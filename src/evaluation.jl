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
    E = [spzeros(Int,length(ordered_observations(m))) for _ in eachindex(pg.nodes)]
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

#PolicyGraphs struct
struct PolicyGraphEvaluator
    depth::Int
    updater::Updater
    eval_tolerance::Float64
    eval_discount::Float64
    use_beliefs::Bool
end

""" 
    PolicyGraphEvaluator(m::POMDP,depth::Int;eval_tolerance::Float64=0.001,eval_discount=discount(m),use_beliefs=true)

    Instantiates a PolicyGraphEvaluator, which evaluates a POMDP policy by building 
    a policy graph (checking for redundant nodes) to some `depth`, until all beliefs 
    are already in the tree, or all beliefs are terminal. Uses `DiscreteUpdater`
    by default. Evaluates this policy graph using iteration to `eval_tolerance`.
    
    Optionally uses beliefs `use_beliefs` used to label nodes in PolicyGraph to improve 
    computational efficiency by iterating over only the states in the belief at each node 
    rather than all states in the POMDP.

    Optionally pass a different `eval_discount` than the default problem discount.
"""
function PolicyGraphEvaluator(m::POMDP,depth::Int;eval_tolerance::Float64=0.001,eval_discount=discount(m),use_beliefs=true)
    return PolicyGraphEvaluator(depth,DiscreteUpdater(m),eval_tolerance,eval_discount,use_beliefs)
end

"""
    evaluate_polgraph(m::POMDP{S,A},s_m::EvalTabularPOMDP,pg::PolicyGraph;tolerance::Float64=0.001,disc=discount(m),use_beliefs::Bool=false)
    evaluate_polgraph(m::POMDP{S,A},pg::PolicyGraph;tolerance::Float64=0.001,rewardfunction=VecReward(),disc=discount(m),use_beliefs::Bool=false)

    Evaluates a PolicyGraph using iteration. Returns a value matrix (with each column corresponding to a state and each row corresponding to a graph node)
    Reward function used for evaluation is that used to create the EvalTabularPOMDP struct.
    
    
    Optionally uses beliefs used to label nodes in PolicyGraph to improve computational efficiency by iterating over only the states in the belief at each node.
    Optionally pass a custom reward function (which must return a vector) to incorporate cost or other functions.

    Modified from: https://jair.org/index.php/jair/article/view/11216
"""
function evaluate_polgraph end
function evaluate_polgraph(m::POMDP{S,A},s_m::EvalTabularPOMDP,pg::PolicyGraph;
    tolerance::Float64=0.001,disc=discount(m),use_beliefs::Bool=false) where {S,A}
    if use_beliefs
        if isempty(pg.beliefs)
            throw("Policy graph belief vector is empty. Either set use_beliefs=false or return beliefs when generating policy graph using store_beliefs=true in gen_polgraph.")
        else
            return evaluate_polgraph_b(m, s_m, pg, pg.beliefs, tolerance, disc)
        end
    else
        return evaluate_polgraph_nb(m, s_m, pg, tolerance, disc)
    end
end

function evaluate_polgraph(m::POMDP{S,A},pg::PolicyGraph;
    tolerance::Float64=0.001,disc=discount(m),use_beliefs::Bool=false,rewardfunction=VecReward()) where {S,A}
    a = first(actions(m))
    s = rand(initialstate(m))
    rew_size = length(rewardfunction(m, s, a))

    s_m = EvalTabularPOMDP(m;rew_f=rewardfunction,r_len = rew_size)
    return evaluate_polgraph(m,s_m,pg;tolerance=tolerance,disc=disc,use_beliefs=use_beliefs)
end

function evaluate_polgraph_nb(m::POMDP{S,A},s_m::EvalTabularPOMDP,pg::PolicyGraph,
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

function evaluate_polgraph_b(m::POMDP{S,A}, s_m::EvalTabularPOMDP, pg::PolicyGraph, 
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
                    @. v_int = @view s_m.R[s_idx,a_idx,:]
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
    gen_eval_polgraph(m::POMDP, updater::Updater, pol::AlphaVectorPolicy, b0::DiscreteBelief, depth::Int; eval_tolerance::Float64=0.001, rewardfunction=VecReward(), use_beliefs::Bool=true, returnpg=false)

    Generates a policy graph using `gen_polgraph` and evaluates it with `eval_polgraph`, returning a value matrix (with each column corresponding to a state and each row corresponding to a graph node) and the policy graph.

    Optionally pass a custom reward function (which must return a vector) to incorporate cost or other functions.
    Optionally uses beliefs used to label nodes in PolicyGraph to improve computational efficiency by iterating over only the states in the belief at each node.
    Optionally return the policy graph.
"""

function gen_eval_polgraph end
function gen_eval_polgraph(m::POMDP{S,A}, updater::Updater, pol::AlphaVectorPolicy,b0::DiscreteBelief, depth::Int; 
    eval_tolerance::Float64=0.001, rewardfunction=VecReward(), disc=discount(m), use_beliefs::Bool=true) where {S,A}
    a = first(actions(m))
    s = rand(initialstate(m))
    rew_size = length(rewardfunction(m, s, a))

    s_m = EvalTabularPOMDP(m;rew_f=rewardfunction,r_len=rew_size)
    pg = gen_polgraph(m, s_m, updater, pol, b0, depth; store_beliefs=use_beliefs)
    values = evaluate_polgraph(m, s_m, pg; tolerance=eval_tolerance, disc=disc, use_beliefs=use_beliefs)

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
    Online value function for a policy on a POMDP given a belief.
    
    Calculates the value of a policy given some belief using a policy graph.
"""
struct PGValueFunction{M<:POMDP,R} <: Function
    m::M
    evaluator::PolicyGraphEvaluator
    pol::Policy
    rewardfunction::R
end

"""
    evaluate(evaluator::PolicyGraphEvaluator, m::POMDP{S,A}, pol::Policy; rewardfunction=VecReward())

    Returns a PGValueFunction, which calculates the value of a belief using a policy graph.
    
    Optionally pass a custom reward function (which must return a vector) to incorporate cost or other functions.
"""
function POMDPTools.evaluate(evaluator::PolicyGraphEvaluator, m::POMDP{S,A}, pol::Policy; rewardfunction=VecReward()) where {S,A}
    @assert isa(evaluator.updater,DiscreteUpdater)
    return PGValueFunction(m,evaluator,pol,rewardfunction)
end

function (v::PGValueFunction)(b0)
    @assert b0.pomdp==v.m

    values,pg = gen_eval_polgraph(v.m, v.evaluator.updater, v.pol, b0, v.evaluator.depth;
    eval_tolerance=v.evaluator.eval_tolerance,
    rewardfunction=v.rewardfunction, disc=v.evaluator.eval_discount, use_beliefs=v.evaluator.use_beliefs)

    return calc_belvalue_polgraph(pg, values, b0)
end

# function evaluate(evaluator::PolicyGraphEvaluator, m::POMDP{S,A}, pol::Policy, b0::DiscreteBelief; rewardfunction=VecReward()) where {S,A}
#     @assert b0.pomdp==m
#     @assert isa(evaluator.updater,DiscreteUpdater)

#     values,pg = gen_eval_polgraph(m, evaluator.updater, pol, b0, evaluator.depth;
#     eval_tolerance=evaluator.eval_tolerance,
#     rewardfunction=rewardfunction, disc=evaluator.eval_discount, use_beliefs=evaluator.use_beliefs)

#     return calc_belvalue_polgraph(pg, values, b0)
# end
