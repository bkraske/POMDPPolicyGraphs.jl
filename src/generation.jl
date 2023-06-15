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

function sparse_recursive_tree(m::POMDP, s_pomdp::EvalTabularPOMDP, updater::Updater, pol::Policy, b0::SparseVector, depth::Int, action_list, edge_list, b_list, d, j_old, a_old)
    if d < depth
        d += 1
        obs = s_pomdp.O[a_old]
        pred = s_pomdp.T[a_old]*b0
        for o in axes(obs,2)
            bp = corrector(s_pomdp, pred, a_old, o)
            if bp ∈ b_list
                push!(edge_list, (j_old, observations(m)[o]) => findall(x->x==bp, b_list)[1])
            else
                po = sum(bp)
                if po > 0.
                    bp.nzval ./= po

                    a = action_from_vec(m, pol, bp)
                    push!(action_list, actions(m)[a])
                    push!(b_list,bp)
                    j = copy(length(action_list))
                    push!(edge_list, (j_old, observations(m)[o]) => j)

                    sparse_recursive_tree(m,s_pomdp,updater,pol,bp,depth,action_list,edge_list,b_list,d,j,a)
                end    
            end
        end
    end
end

function sparse_recursive_tree(m::POMDP{S,A}, updater::Updater, pol::Policy, b0::DiscreteBelief, depth::Int; replace::Vector=[]) where {S,A}
    edge_list = Dict{Tuple{Int64,obstype(pol.pomdp)},Int64}()
    action_list = A[]
    b_list = SparseVector{Float64, Int64}[]
    s_pomdp = EvalTabularPOMDP(m)
    d = 0
    a=if !isempty(replace)
        replace[1]
    else
        action(pol, b0)
    end::A
    push!(action_list, a)
    push!(b_list,sparse(b0.b))
    j = copy(length(action_list))

    sparse_recursive_tree(m, s_pomdp, updater, pol, sparse(b0.b), depth, action_list, edge_list, b_list, d, j, actionindex(m,a))

    return PolicyGraph(action_list, edge_list, 1)
end


function recursive_tree(m::POMDP, updater::Updater, pol::Policy, b0::DiscreteBelief, depth::Int, action_list, edge_list, d, j_old, a_old)
    if d < depth
        d += 1
        for o in observations(m)
            if is_nonzero_obs(m, a_old, b0, o)
                bp = update(updater, b0, a_old, o)
                a = action(pol, bp)
                push!(action_list, a)
                j = copy(length(action_list))
                push!(edge_list, (j_old, o) => j)
                recursive_tree(m,updater,pol,bp,depth,action_list,edge_list,d,j,a)
            end
        end
    end
end

function recursive_tree(m::POMDP{S,A}, updater::Updater, pol::Policy, b0::DiscreteBelief, depth::Int; replace::Vector=[]) where {S,A}
    edge_list = Dict{Tuple{Int64,obstype(pol.pomdp)},Int64}()
    action_list = A[]
    d = 0
    a=if !isempty(replace)
        replace[1]
    else
        action(pol, b0)
    end::A
    push!(action_list, a)
    j = copy(length(action_list))

    recursive_tree(m::POMDP, updater::Updater, pol::Policy, b0::DiscreteBelief, depth::Int, action_list, edge_list, d, j, a)

    return PolicyGraph(action_list, edge_list, 1)
end


function policy_tree(m::POMDP{S,A}, updater::Updater, pol::Policy, b0::DiscreteBelief, depth::Int; replace::Vector=[]) where {S,A}
    edge_list = Dict{Tuple{Int64,obstype(pol.pomdp)},Int64}()
    action_list = A[]
    node_list = Int[]
    j = 1
    queue = [(b0, 0, j)]
    num_outer = 0
    while !isempty(queue)
        num_nnz = 0
        num_outer += 1
        b, d, i = popfirst!(queue)
        a=if j == 1 && !isempty(replace)
            replace[1]
            # @show a
        else
            action(pol, b)
        end::A
        push!(action_list, a)
        push!(node_list, i)
        if d < depth
            for o in observations(m)
                # @show length(observations(m))
                if is_nonzero_obs(m, a, b, o)
                    num_nnz += 1
                    j += 1
                    bp = update(updater, b, a, o)
                    push!(queue, (bp, d + 1, j))
                    push!(edge_list, (i, o) => j)
                end
            end
        end
        # @show b,d,i
        # @show num_nnz
        # if num_outer >10
        #     break
        # end
    end
    # @show num_outer
    # @show GrzesPolicyGraph(node_list, action_list, edge_list, 1)
    return GrzesPolicyGraph(node_list, action_list, edge_list, 1)
end

function equivalent_cp(m::POMDP, n1::Int, n2::Int, pg::GrzesPolicyGraph)
    if pg.actions[n1] != pg.actions[n2]
        return false
    end
    for o in observations(m)
        if haskey(pg.edges, (n1, o)) 
            #check if the observation corresponds to an edge on node 1
            if !haskey(pg.edges,(n2,o)) 
                #check if the observation corresponds to an edge on node 2
                return false
            elseif !equivalent_cp(m, pg.edges[(n1, o)], pg.edges[(n2, o)], pg)
                return false
            end
        elseif haskey(pg.edges, (n2, o)) 
            #check if the observation corresponds to an edge on node 2 if it doesn't on n1
            return false
        end
    end
    return true
end

function pg_children!(children::Vector, m::POMDP, node::Int, pg::GrzesPolicyGraph)
    for o in observations(m)
        if haskey(pg.edges, (node, o))
            new_node = pg.edges[(node, o)]
            if new_node ∉ children
                push!(children, new_node)
                pg_children!(children, m, new_node, pg)
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

function tree2pg(tree::GrzesPolicyGraph)
    return PolicyGraph(tree.actions, tree.edges, tree.node1)
end

function policy2fsc(m::POMDP, updater::Updater, pol::Policy, b0::DiscreteBelief, depth::Int; replace=[])
    # println("Build Tree")
    pg = policy_tree(m, updater, pol, b0, depth; replace=replace)
    # @show length(pg.nodes)
    # println("Condense Tree")
    node_l = length(pg.nodes)
    big_rm_list = Int64[]
    for n_i in 1:node_l #length(pg.edges)
        # @show n_i
        if n_i ∈ pg.nodes #&& n_i ∉ big_rm_list
            for n_j in 1:(n_i-1)#1:node_l #length(pg.edges)
                # @show n_j
                if n_j ∈ pg.nodes #&& n_j ∉ big_rm_list #n_j < n_i && n_j ∈ pg.nodes
                    if equivalent_cp(m, n_i, n_j, pg)
                        nodes_rm = [n_i]
                        pg_children!(nodes_rm, m, n_i, pg)
                        # deleteat!(pg.nodes, findall(x -> x ∈ nodes_rm, pg.nodes))
                        # append!(big_rm_list,nodes_rm)
                        filter!(x -> x ∉ nodes_rm, pg.nodes)
                        # deleteat!(pg.nodes, pg.nodes .∈ nodes_rm)
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
    # filter!(x -> x ∉ big_rm_list, pg.nodes)
    # println("Convert Tree")
    npg = gpg2pg(pg)
    # r_mat = reward_matrix(SparseTabularPOMDP(m))
    # r_max = maximum(r_mat)
    # r_min = minimum(r_mat)
    # loss = discount(m)^depth * (r_max - r_min) / (1 - discount(m))
    # @info "Initial Belief Value Loss is $loss"
    return npg
end

function policy2tree(m::POMDP, updater::Updater, pol::Policy, b0::DiscreteBelief, depth::Int)
    pg = policy_tree(m, updater, pol, b0, depth)
    npg = tree2pg(pg)
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