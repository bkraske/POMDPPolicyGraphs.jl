##Structs
struct PolicyGraph{N,E} <: Policy
    nodes::N
    edges::E
    node1::Int64
end

struct GrzesPolicyGraph{N,E} <: Policy
    nodes::Vector{Int}
    actions::N
    edges::E
    node1::Int64
end

struct CGCPPolicyGraph{N,E} <: Policy
    nodes::Vector{Tuple{Int64,Int64}}
    actions::N
    edges::E
    node1::Tuple{Int64,Int64}
end

##Utility Functions
"""
    is_nonzero_obs(pol::Policy,a,b::DiscreteBelief,o)

    Checks if observation probability is zero a given a policy, action, belief, and observation

"""
function is_nonzero_obs end

function is_nonzero_obs(pomdp::POMDP, a, b::DiscreteBelief, o)
    optot = 0.0
    for (i, obs_prob) in enumerate(b.b)
        if obs_prob > 0
            s = b.state_list[i]
            td = transition(pomdp, s, a)
            for (sp, tp) in weighted_iterator(td)
                if tp > 0
                    op = obs_weight(pomdp, s, a, sp, o) # shortcut for observation probability from POMDPModelTools
                elseif tp == 0
                    op = 0.0
                end
                optot += op
            end
        end
    end
    return optot > 0
end

function is_nonzero_obs(pomdp::POMDP, a, b::SparseVector{Float64, Int64}, o)
    optot = 0.0
    for (i, obs_prob) in enumerate(b)
        if obs_prob > 0
            s = ordered_states(pomdp)[i]
            td = transition(pomdp, s, a)
            for (sp, tp) in weighted_iterator(td)
                if tp > 0
                    op = obs_weight(pomdp, s, a, sp, o) # shortcut for observation probability from POMDPModelTools
                elseif tp == 0
                    op = 0.0
                end
                optot += op
            end
        end
    end
    return optot > 0
end

##Grzes Methods
function policy_tree(m::POMDP{S,A}, updater::Updater, pol::Policy, b0::DiscreteBelief, depth::Int) where {S,A}
    edge_list = Dict{Tuple{Int64,obstype(pol.pomdp)},Int64}()
    action_list = A[]
    node_list = Int[]
    j = 1
    queue = [(b0, 0, j)]
    while !isempty(queue)
        b, d, i = popfirst!(queue)
        a = action(pol, b)
        push!(action_list, a)
        push!(node_list, i)
        if d < depth
            for o in observations(m)
                if is_nonzero_obs(m, a, b, o)
                    j += 1
                    bp = update(updater, b, a, o)
                    push!(queue, (bp, d + 1, j))
                    push!(edge_list, (i, o) => j)
                end
            end
        end
    end
    # @show GrzesPolicyGraph(node_list, action_list, edge_list, 1)
    return GrzesPolicyGraph(node_list, action_list, edge_list, 1)
end

function equivalent_cp(m::POMDP, n1::Int, n2::Int, pg::GrzesPolicyGraph)
    if pg.actions[n1] != pg.actions[n2]
        return false
    end
    for o in observations(m)
        # if haskey(pg.edges,(n1,o)) && !haskey(pg.edges,(n2,o))
        #     return false
        # else
        @show (n1, o)
        @show pg.edges[(n1, o)]
        @show (n2, o)
        @show pg.edges[(n2, o)]
        if haskey(pg.edges, (n1, o)) && !equivalent_cp(m, pg.edges[(n1, o)], pg.edges[(n2, o)], pg)
            return false
        end
    end
    return true
end

function pg_children!(children::Vector, m::POMDP, node::Int, pg::GrzesPolicyGraph)
    for o in observations(m)
        if haskey(pg.edges, (node, o))
            if pg.edges[(node, o)] ∉ children
                push!(children, pg.edges[(node, o)])
                pg_children!(children, m, pg.edges[(node, o)], pg)
            end
        end
    end
end

function clean_gpg(pg::GrzesPolicyGraph)
    acts = Dict(pg.nodes .=> pg.actions[pg.nodes])
    edges = pg.edges
    for (k, v) in edges
        if k[1] ∉ pg.nodes
            delete!(pg.edges, k)
        end
    end
    return GrzesPolicyGraph(pg.nodes, acts, edges, 1)
end

function gpg2pg(pg::GrzesPolicyGraph)
    o2n = Dict(pg.nodes .=> 1:length(pg.nodes))
    action_list = []
    for n in pg.nodes
        push!(action_list, pg.actions[n])
    end
    edge_list = Dict()
    for (k, v) in pg.edges
        n1 = k[1]
        o = k[2]
        if n1 ∈ pg.nodes || v ∈ pg.nodes
            push!(edge_list, (o2n[n1], o) => o2n[v])
        end
    end
    return PolicyGraph(action_list, edge_list, 1)
end

function policy2fsc(m::POMDP, updater::Updater, pol::Policy, b0::DiscreteBelief, depth::Int)
    pg = policy_tree(m, updater, pol, b0, depth)
    for n_i in 1:length(pg.edges)
        if n_i ∈ pg.nodes
            for n_j in 1:length(pg.edges)
                if n_j ∈ pg.nodes && n_j < n_i
                    if equivalent_cp(m, n_i, n_j, pg)
                        nodes_rm = [n_i]
                        pg_children!(nodes_rm, m, n_i, pg)
                        deleteat!(pg.nodes, findall(x -> x ∈ nodes_rm, pg.nodes))
                        for n in 1:length(pg.actions), o in observations(m)
                            if haskey(pg.edges, (n, o)) && pg.edges[(n, o)] == n_i
                                push!(pg.edges, (n, o) => n_j)
                            end
                        end
                    end
                end
            end
        end
    end
    npg = gpg2pg(pg)
    # r_mat = reward_matrix(SparseTabularPOMDP(m))
    # r_max = maximum(r_mat)
    # r_min = minimum(r_mat)
    # loss = discount(m)^depth * (r_max - r_min) / (1 - discount(m))
    # @info "Initial Belief Value Loss is $loss"
    return npg
end

# function alpha_2_fsc(m::POMDP,updater::Updater,pol::AlphaVectorPolicy,b0::DiscreteBelief)
#     edge_list = Dict{Tuple{Int64,obstype(pol.pomdp)},Int64}()
#     a_to_n = Dict{Int,Int}()
#     node_list = A[]
#     node_counter = [1]
#     a0, n0 = dot_α(node_list,b0,pol,a_to_n,node_counter)
#     node_list = action_map
#     for i in 1:length(pol.alphas)
#         for o in observations(m)
#             if is_nonzero_obs(m,a,b,o) #NEED witness beliefs
#                 push!(edge_list, (n,o) => np)
#             else
#                 push!(edge_list, (n,o) => np)
#             end
#         end
#     end
#     return PolicyGraph(node_list, edge_list, n0)
# end

##CGCP Witness Method
function max_alpha_val_ind(Γ, b::DiscreteBelief)
    max_ind = 1
    max_val = -Inf
    for (i, α) ∈ enumerate(Γ)
        val = dot(α, b.b)
        if val > max_val
            max_ind = i
            max_val = val
        end
    end
    return max_ind
end

function max_alpha_val_ind(Γ, b::SparseVector{Float64,Int64})
    max_ind = 1
    max_val = -Inf
    for (i, α) ∈ enumerate(Γ)
        val = dot(α, b)
        if val > max_val
            max_ind = i
            max_val = val
        end
    end
    return max_ind
end

function CGCP_pg(m::POMDP{S,A}, updater::Updater, pol::AlphaVectorPolicy, beliefs, depths::Vector{Int}) where {S,A}
    edge_list = Dict{Tuple{Tuple{Int64,Int64},obstype(pol.pomdp)},Tuple{Int64,Int64}}()
    action_list = A[]
    node_list = Tuple{Int64,Int64}[]
    h = maximum(depths)
    for t in h:-1:0
        inds = findall(x->x==t,depths)
        # @show pol.alphas[inds] .=> pol.action_map[inds]
        Γt = pol.alphas[inds]
        indstp1 = findall(x->x==t+1,depths)
        # @show  pol.alphas[indstp1] .=> pol.action_map[indstp1]
        Γtp1 = pol.alphas[indstp1]
        a_lst = pol.action_map[inds]
        b_lst = beliefs[inds]
        # println("Depth $t==================")
        for j in 1:length(Γt)
            # println("Index $j, action $(a_lst[j])==================")
            push!(node_list, (t, j))
            a = a_lst[j]
            push!(action_list, a)
            # @show a
            b = DiscreteBelief(m,Vector(b_lst[j]))
            # @show b
            if t < h
                for o in observations(m)
                    # println("Observation $o")
                    if is_nonzero_obs(m, a, b, o)
                        bp = update(updater, b, a, o)
                        # @show bp
                        k = max_alpha_val_ind(Γtp1, bp)
                        # @show Γtp1
                        # @show pol.action_map[indstp1]
                        # @show pol.action_map[indstp1][k]
                        # @show ((t, j), o) => (t + 1, k)
                        push!(edge_list, ((t, j), o) => (t + 1, k))
                    else
                        push!(edge_list, ((t, j), o) => (t + 1, 1))
                    end
                end
            end
        end
    end
    ind = findall(x->x==0,depths)
    @assert length(ind)==1
    node1 = max_alpha_val_ind(pol.alphas[ind], beliefs[ind...])
    return CGCPPolicyGraph(node_list, action_list, edge_list, (0, node1))
end

function CGCP2PG(pg::CGCPPolicyGraph)
    o2n = Dict(pg.nodes .=> 1:length(pg.nodes))
    action_list = pg.actions
    edge_list = Dict()
    for (k, v) in pg.edges
        n1 = k[1]
        o = k[2]
        if n1 ∈ pg.nodes || v ∈ pg.nodes
            push!(edge_list, (o2n[n1], o) => o2n[v])
        else
            throw("$n1 or $v not in node list")
        end
    end
    return PolicyGraph(action_list, edge_list, o2n[pg.node1])
end


function CGCP_pg2(m::POMDP{S,A}, updater::Updater, pol::AlphaVectorPolicy, beliefs, depths::Vector{Int}) where {S,A}
    edge_list = Dict{Tuple{Tuple{Int64,Int64},obstype(pol.pomdp)},Tuple{Int64,Int64}}()
    action_list = A[]
    node_list = Tuple{Int64,Int64}[]
    h = maximum(depths)
    Γ = pol.alphas
    a_lst = pol.action_map
    b_lst = beliefs
    for t in 0:1:h #h:-1:0
        for j in 1:length(Γ)
            # println("Index $j, action $(a_lst[j])==================")
            push!(node_list, (t, j))
            a = a_lst[j]
            push!(action_list, a)
            # @show a
            b = DiscreteBelief(m,Vector(b_lst[j]))
            # @show b
            if t < h
                for o in observations(m)
                    # println("Observation $o")
                    if is_nonzero_obs(m, a, b, o)
                        # @show b
                        # @show o
                        bp = update(updater, b, a, o)
                        # @show bp
                        k = max_alpha_val_ind(Γ, bp)
                        # @show Γtp1
                        # @show pol.action_map[indstp1]
                        # @show k
                        # @show pol.action_map[k]
                        # @show ((t, j), o) => (t + 1, k)
                        push!(edge_list, ((t, j), o) => (t + 1, k))
                    else
                        push!(edge_list, ((t, j), o) => (t + 1, 1))
                    end
                end
            end
        end
    end
    ind = findall(x->x==0,depths)
    @assert length(ind)==1
    node1 = max_alpha_val_ind(pol.alphas[ind], beliefs[ind...])
    return CGCPPolicyGraph(node_list, action_list, edge_list, (0, node1))
end

function equivalent_cp(m::POMDP, n1::Int, n2::Int, pg)
    if pg.nodes[n1] != pg.nodes[n2]
        return false
    end
    for o in observations(m)
        # if haskey(pg.edges,(n1,o)) && !haskey(pg.edges,(n2,o))
        #     return false
        # else
        if haskey(pg.edges, (n1, o)) && !equivalent_cp(m, pg.edges[(n1, o)], pg.edges[(n2, o)], pg)
            return false
        end
    end
    return true
end

function pg_children!(children::Vector, m::POMDP, node::Int, pg)
    for o in observations(m)
        if haskey(pg.edges, (node, o))
            if pg.edges[(node, o)] ∉ children
                push!(children, pg.edges[(node, o)])
                pg_children!(children, m, pg.edges[(node, o)], pg)
            end
        end
    end
end

function policy2fsc(m::POMDP, pg)
    pg = deepcopy(pg)
    for n_i in 1:length(pg.edges)
        if n_i ∈ pg.nodes
            for n_j in 1:length(pg.edges)
                if n_j ∈ pg.nodes && n_j < n_i
                    if equivalent_cp(m, n_i, n_j, pg)
                        nodes_rm = [n_i]
                        pg_children!(nodes_rm, m, n_i, pg)
                        deleteat!(pg.nodes, findall(x -> x ∈ nodes_rm, pg.nodes))
                        for n in 1:length(pg.actions), o in observations(m)
                            if haskey(pg.edges, (n, o)) && pg.edges[(n, o)] == n_i
                                push!(pg.edges, (n, o) => n_j)
                            end
                        end
                    end
                end
            end
        end
    end
    return pg
end