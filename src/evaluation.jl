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
    eval_polgraph(m::POMDP{S,A},s_m::EvalTabularPOMDP,pg::PolicyGraph;tolerance::Float64=0.001,disc=discount(m),use_beliefs::Bool=false)
    eval_polgraph(m::POMDP{S,A},pg::PolicyGraph;tolerance::Float64=0.001,rewardfunction=VecReward(),disc=discount(m),use_beliefs::Bool=false)

    Evaluates a PolicyGraph using iteration. Returns a value matrix (with each column corresponding to a state and each row corresponding to a graph node)
    Reward function used for evaluation is that used to create the EvalTabularPOMDP struct.
    
    
    Optionally uses beliefs used to label nodes in PolicyGraph to improve computational efficiency by iterating over only the states in the belief at each node.
    Optionally pass a custom reward function (which must return a vector) to incorporate cost or other functions.

    Modified from: https://jair.org/index.php/jair/article/view/11216
"""
function eval_polgraph end
function eval_polgraph(m::POMDP{S,A},s_m::EvalTabularPOMDP,pg::PolicyGraph;
    tolerance::Float64=0.001,disc=discount(m),use_beliefs::Bool=false) where {S,A}
    if use_beliefs
        if isempty(pg.beliefs)
            throw("Policy graph belief vector is empty. Either set use_beliefs=false or return beliefs when generating policy graph using store_beliefs=true in gne_polgraph.")
        else
            return eval_polgraph_b(m, s_m, pg, pg.beliefs, tolerance, disc)
        end
    else
        return eval_polgraph_nb(m, s_m, pg, tolerance, disc)
    end
end

function eval_polgraph(m::POMDP{S,A},pg::PolicyGraph;
    tolerance::Float64=0.001,disc=discount(m),use_beliefs::Bool=false,rewardfunction=VecReward()) where {S,A}
    a = first(actions(m))
    s = rand(initialstate(m))
    rew_size = length(rewardfunction(m, s, a))

    s_m = EvalTabularPOMDP(m;rew_f=rewardfunction,r_len = rew_size)
    return eval_polgraph(m,s_m,pg;tolerance=tolerance,disc=disc,use_beliefs=use_beliefs)
end

function eval_polgraph_nb(m::POMDP{S,A},s_m::EvalTabularPOMDP,pg::PolicyGraph,
    tolerance::Float64,disc) where {S,A}
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
                    @. v_int = s_m.R[s_idx,a_idx,:]::Vector{Float64}
                    t_dist::SparseArrays.SparseVector{Float64, Int64} = @view s_m.T[a_idx][:,s_idx]
                    for sp_idx in SparseArrays.nonzeroinds(t_dist)
                        prob_t = t_dist[sp_idx]::Float64
                        for o_idx in SparseArrays.nonzeroinds(s_edges[i])
                            prob_o = s_m.O2[a_idx][o_idx,sp_idx]::Float64
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

function eval_polgraph_b(m::POMDP{S,A}, s_m::EvalTabularPOMDP, pg::PolicyGraph, 
    b_list::Vector{SparseArrays.SparseVector{Float64, Int64}},tolerance::Float64,disc) where {S,A}
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

    s_edges = edge_dict_to_array(m,pg)

    while norm(diff_mat .= v_p .- v, Inf) > tolerance
        count += 1
        v .= v_p
        for i in eachindex(pg.nodes)
            for s_idx in SparseArrays.nonzeroinds(b_list[i])
                if !s_m.isterminal[s_idx]
                    a = pg.nodes[i]::A
                    a_idx = actionindex(m,a)
                    @. v_int = s_m.R[s_idx,a_idx,:]::Vector{Float64}
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
    gen_eval_polgraph(m::POMDP, updater::Updater, pol::AlphaVectorPolicy, b0::DiscreteBelief, depth::Int; eval_tolerance::Float64=0.001, rewardfunction=VecReward(), replace=[], use_beliefs::Bool=true, returnpg=false)

    Generates a policy graph using `gen_polgraph` and evaluates it with `eval_polgraph`, returning a value matrix (with each column corresponding to a state and each row corresponding to a graph node) and the policy graph.

    Optionally pass a custom reward function (which must return a vector) to incorporate cost or other functions.
    Optionally replace the first action in the Policy Graph with an alternative action, e.g. `replace=[:up]`
    Optionally uses beliefs used to label nodes in PolicyGraph to improve computational efficiency by iterating over only the states in the belief at each node.
    Optionally return the policy graph.
"""

function gen_eval_polgraph end
function gen_eval_polgraph(m::POMDP{S,A}, updater::Updater, pol::AlphaVectorPolicy,b0::DiscreteBelief, depth::Int; 
    eval_tolerance::Float64=0.001, rewardfunction=VecReward(), disc=discount(m), replace=A[], use_beliefs::Bool=true) where {S,A}
    a = first(actions(m))
    s = rand(initialstate(m))
    rew_size = length(rewardfunction(m, s, a))

    s_m = EvalTabularPOMDP(m;rew_f=rewardfunction,r_len=rew_size)
    pg = gen_polgraph(m, s_m, updater, pol, b0, depth; store_beliefs=use_beliefs, replace=replace)
    values = eval_polgraph(m, s_m, pg; tolerance=eval_tolerance, disc=disc, use_beliefs=use_beliefs)

    return values, pg
end


"""
    calc_belvalue_polgraph(pg::PolicyGraph, result::Array, b::DiscreteBelief)

    Takes Policy Graph, Value Vector, and DiscreteBelief. Returns value of initial belief using the state values of the first node in the graph.
"""
function calc_belvalue_polgraph end

function calc_belvalue_polgraph(pg::PolicyGraph, result::Array, b::DiscreteBelief)
    i = pg.node1
    first_node = result[i, :, :]
    if length(support(b)) == size(first_node)[1]
        val_mat = b.b' * first_node
        return val_mat'
    else
        throw("Belief and result columns are different
              sizes: $(length(support(b))), $(size(first_node)[1])")
    end
end
##Get value from belief and state values
"""
    belief_value_polgraph(m::POMDP, updater::Updater, pol::AlphaVectorPolicy, b0::DiscreteBelief, depth::Int; eval_tolerance::Float64=0.001, rewardfunction=VecReward(), replace=[], beliefbased=true)

    Returns value of initial belief using the state values of the first node in the graph.
    
    Optionally pass a custom reward function (which must return a vector) to incorporate cost or other functions.
    Optionally uses beliefs used to label nodes in PolicyGraph to improve computational efficiency by iterating over only the states in the belief at each node.
    Optionally replace the first action in the Policy Graph with an alternative action, e.g. `replace=[:up]`
"""
function belief_value_polgraph end

function belief_value_polgraph(m::POMDP{S,A}, updater::Updater, pol::AlphaVectorPolicy, 
            b0::DiscreteBelief, depth::Int; eval_tolerance::Float64=0.001, 
            rewardfunction=VecReward(), disc=discount(m), replace=A[], use_beliefs=true) where {S,A}
    
    values,pg = gen_eval_polgraph(m, updater, pol, b0, depth;
        eval_tolerance=eval_tolerance,
        rewardfunction=rewardfunction, disc=disc, replace=replace, use_beliefs=use_beliefs)

    return calc_belvalue_polgraph(pg, values, b0)
end