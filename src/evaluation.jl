##Based on the following work:
#https://www.jair.org/index.php/jair/article/view/11216/26427,
#https://www.aaai.org/Papers/AAAI/2008/AAAI08-167.pdf,
#https://www.jair.org/index.php/jair/article/view/11216/26427

##Exact Evaluation with Iteration
"""
    vectorizedReward(m::POMDP,s,a)
    vectorizedReward(m::POMDP,s,a,sp)

    Function for handling default POMDP rewards which are in vector form
"""

function vectorizedReward end
function vectorizedReward(m::POMDP, s, a)
    return [POMDPs.reward(m, s, a)]
end

function vectorizedReward(m::POMDP, s, a, sp)
    return [POMDPs.reward(m, s, a, sp)]
end

#Tyler
"""
    Non-allocating version of `vectorizedReward`
"""
struct VecReward{T}
    dest::Vector{T}
end

VecReward() = VecReward(Vector{Float64}(undef, 1))

function (r::VecReward)(m, s, a, sp)
    return r.dest .= POMDPs.reward(m, s, a, sp)
end

function (r::VecReward)(m, s, a)
    return r.dest .= POMDPs.reward(m, s, a)
end

#Convert dictionary to vector for speed
function edge_dict_to_array(m,pg)
    E = [sparse(Int.(zeros(length(ordered_observations(m))))) for _ in eachindex(pg.nodes)]
    for node in eachindex(pg.nodes)
        for (i,o) in enumerate(ordered_observations(m))
            new_node = get(pg.edges, (node, o), nothing)
            if !isnothing(new_node)
                E[node][i] = new_node
            end
        end
    end
    return E
end

"""
    sparse_eval_pg(m::POMDP{S,A},s_m::EvalTabularPOMDP,pg::PolicyGraph;tolerance::Float64=0.001,disc=discount(m))
    sparse_eval_pg(m::POMDP{S,A},s_m::EvalTabularPOMDP,pg::PolicyGraph,b_list::Vector{SparseArrays.SparseVector{Float64, Int64}};tolerance::Float64=0.001,disc=discount(m))

    Evaluates a PolicyGraph using iteration. Returns a value matrix (with each column corresponding to a state and each row corresponding to a graph node)
    Reward function used for evaluation is that used to create the EvalTabularPOMDP struct.

    Optionally takes beliefs used to label nodes in PolicyGraph to improve computational efficiency by iterating over only the states in the belief at each node.

    Modified from: https://jair.org/index.php/jair/article/view/11216
"""
function sparse_eval_pg end

function sparse_eval_pg(
    m::POMDP{S,A},
    s_m::EvalTabularPOMDP,
    pg::PolicyGraph;
    tolerance::Float64=0.001,
    disc=discount(m)
) where {S,A}
    γ = disc

    Nn = length(pg.nodes)
    Ns = length(states(m))
    rew_size = size(s_m.R,3)

    v = ones(Nn, Ns, rew_size)
    v_p = zeros(Nn, Ns, rew_size)
    diff_mat = Array{Float64,3}(undef, Nn, Ns, rew_size)
    v_int = Vector{Float64}(undef, rew_size)
    v_tmp = copy(v_int)

    count = 0

    os = ordered_states(m)

    s_edges = edge_dict_to_array(m,pg)

    while norm(diff_mat .= v_p .- v, Inf) > tolerance
        count += 1
        v .= v_p
        for i in eachindex(pg.nodes)
            for s_idx in eachindex(os)
                if !s_m.isterminal[s_idx]
                    a = pg.nodes[i]::A
                    a_idx = actionindex(m,a)
                    @. v_int = s_m.R[s_idx,a_idx]
                    t_dist = @view s_m.T[a_idx][:,s_idx]
                    for sp_idx in SparseArrays.nonzeroinds(t_dist)
                        prob_t = t_dist[sp_idx]
                        for o_idx in SparseArrays.nonzeroinds(s_edges[i])
                            prob_o = s_m.O2[a_idx][o_idx,sp_idx]
                            node = s_edges[i][o_idx]
                            @inbounds copyto!(v_tmp, @view v[node::Int, sp_idx, :])
                            @. v_int += (v_tmp *= γ * prob_t * prob_o)
                        end
                    end
                    @inbounds copyto!(view(v_p, i, s_idx, :), v_int)
                end
            end
        end
    end
    return v_p
end

function sparse_eval_pg(
    m::POMDP{S,A},
    s_m::EvalTabularPOMDP,
    pg::PolicyGraph,
    b_list::Vector{SparseArrays.SparseVector{Float64, Int64}};
    tolerance::Float64=0.001,
    disc=discount(m)) where {S,A}
    γ = disc

    Nn = length(pg.nodes)
    Ns = length(states(m))
    rew_size = size(s_m.R,3)

    v = ones(Nn, Ns, rew_size)
    v_p = zeros(Nn, Ns, rew_size)
    diff_mat = Array{Float64,3}(undef, Nn, Ns, rew_size)
    v_int = Vector{Float64}(undef, rew_size)
    v_tmp = copy(v_int)

    count = 0

    os = ordered_states(m)

    s_edges = edge_dict_to_array(m,pg)

    while norm(diff_mat .= v_p .- v, Inf) > tolerance
        count += 1
        v .= v_p
        for i in eachindex(pg.nodes)
            for s_idx in SparseArrays.nonzeroinds(b_list[i])
                if !s_m.isterminal[s_idx]
                    a = pg.nodes[i]::A
                    a_idx = actionindex(m,a)
                    @. v_int = s_m.R[s_idx,a_idx]
                    t_dist = @view s_m.T[a_idx][:,s_idx]
                    for sp_idx in SparseArrays.nonzeroinds(t_dist)
                        prob_t = t_dist[sp_idx]
                        for o_idx in SparseArrays.nonzeroinds(s_edges[i])
                            prob_o = s_m.O2[a_idx][o_idx,sp_idx]
                            node = s_edges[i][o_idx]
                            @inbounds copyto!(v_tmp, @view v[node::Int, sp_idx, :])
                            @. v_int += (v_tmp *= γ * prob_t * prob_o)
                        end
                    end
                    @inbounds copyto!(view(v_p, i, s_idx, :), v_int)
                end
            end
        end
    end
    return v_p
end

##Convenience Functions
"""
    gen_eval_pg(m::POMDP, updater::Updater, pol::AlphaVectorPolicy, b0::DiscreteBelief, depth::Int, eval_tolerance::Float64=0.001, rewardfunction=VecReward(), replace=[], beliefbased=true, returnpg=false)

    Generates a policy graph using `sparse_recursive_tree` and evaluates it with `sparse_eval_pg`, returning a value matrix  (with each column corresponding to a state and each row corresponding to a graph node).

    Optionally pass a custom reward function (which must return a vector) to incorporate cost or other functions.
    Optionally replace the first action in the Policy Graph with an alternative action, e.g. `replace=[:up]`
    Optionally uses beliefs used to label nodes in PolicyGraph to improve computational efficiency by iterating over only the states in the belief at each node.
    Optionally return the policy graph.
"""

function gen_eval_pg end
function gen_eval_pg(m::POMDP, updater::Updater, pol::AlphaVectorPolicy, 
            b0::DiscreteBelief, depth::Int; 
            eval_tolerance::Float64=0.001, rewardfunction=VecReward(), replace=[], beliefbased=true, returnpg=false)
    a = rand(actions(m))
    s = rand(initialstate(m))
    rew_size = length(rewardfunction(m, s, a))

    s_m = EvalTabularPOMDP(m;rew_f=rewardfunction,r_len = rew_size)

    if beliefbased == false
        pg = sparse_recursive_tree(m, s_m, updater, pol, b0, depth;replace=replace)
        values = sparse_eval_pg(m, s_m, pg; tolerance=eval_tolerance)
    else
        pg,bels = sparse_recursive_tree(m, s_m, updater, pol, b0, depth;replace=replace,return_bels=true)
        values = sparse_eval_pg(m, s_m, pg, bels; tolerance=eval_tolerance)
    end

    if !returnpg
        return values
    else
        return (values, pg)
    end
end


"""
    get_belief_value(pg::PolicyGraph, result::Array, b::DiscreteBelief)

    Takes Policy Graph, Value Vector, and DiscreteBelief. Returns value of initial belief using the state values of the first node in the graph.
"""
function get_belief_value end

function get_belief_value(pg::PolicyGraph, result::Array, b::DiscreteBelief)
    i = pg.node1
    first_node = result[i, :, :]
    if length(support(b)) == size(first_node)[1]
        return b.b' * first_node
    else
        throw("Belief and result columns are different
              sizes: $(length(support(b))), $(size(first_node)[1])")
    end
end
##Get value from belief and state values
"""
    gen_belief_value(m::POMDP, updater::Updater, pol::AlphaVectorPolicy, b0::DiscreteBelief, depth::Int; eval_tolerance::Float64=0.001, rewardfunction=VecReward(), replace=[], beliefbased=true)

    Returns value of initial belief using the state values of the first node in the graph.
    Optionally uses beliefs used to label nodes in PolicyGraph to improve computational efficiency by iterating over only the states in the belief at each node.
    Optionally replace the first action in the Policy Graph with an alternative action, e.g. `replace=[:up]`
"""
function gen_belief_value end

function gen_belief_value(m::POMDP, updater::Updater, pol::AlphaVectorPolicy, 
            b0::DiscreteBelief, depth::Int; eval_tolerance::Float64=0.001, 
            rewardfunction=VecReward(), replace=[], beliefbased=true)
    
    if beliefbased == false
        values,pg = gen_eval_pg(m, updater, pol, b0, depth;eval_tolerance=eval_tolerance, rewardfunction=rewardfunction, 
                                replace=replace, beliefbased=beliefbased, returnpg=true)
    else
        values,pg = gen_eval_pg(m, updater, pol, b0, depth;eval_tolerance=eval_tolerance, rewardfunction=rewardfunction, 
                                replace=replace, beliefbased=beliefbased, returnpg=true)
    end

    return get_belief_value(pg, values, b0)
end