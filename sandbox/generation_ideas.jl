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

function policy_tree_pg(m::POMDP{S,A}, updater::Updater, pol::Policy, b0::DiscreteBelief, depth::Int) where {S,A}
    edge_list = Dict{Tuple{Int64,obstype(pol.pomdp)},Int64}()
    action_list = A[]
    j = 1
    queue = [(b0, 0, j)]
    num_outer = 0
    while !isempty(queue)
        num_nnz = 0
        num_outer += 1
        b, d, i = popfirst!(queue)
        a = action(pol, b)
        push!(action_list, a)
        if d < depth
            for o in observations(m)
                if is_nonzero_obs(m, a, b, o)
                    num_nnz += 1
                    j += 1
                    bp = update(updater, b, a, o)
                    push!(queue, (bp, d + 1, j))
                    push!(edge_list, (i, o) => j)
                end
            end
        end
    end
    return PolicyGraph(action_list, edge_list, 1)
end

function TreeBeliefValue(m::POMDP, updater::Updater, pol::AlphaVectorPolicy, 
    b0::DiscreteBelief, depth::Int;
    eval_tolerance::Float64=0.001, rewardfunction=VecReward())
    @show rewardfunction
    println("Generate PG")
    pg = policy_tree_pg(m, updater, pol, b0, depth)
    println("Evaluate PG")
    values = EvalPolicyGraph(m, pg; tolerance=eval_tolerance, rewardfunction=rewardfunction)
    i = pg.node1
    first_node = values[i, :, :]
    if length(support(b0)) == size(first_node)[1]
        return b0.b' * first_node
    else
        throw("Belief and result columns are different
        sizes: $(length(support(b))), $(size(first_node)[1])")
    end
end