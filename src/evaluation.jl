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
function vectorizedReward(m::POMDP,s,a)
    return [POMDPs.reward(m,s,a)]
end

function vectorizedReward(m::POMDP,s,a,sp)
    return [POMDPs.reward(m,s,a,sp)]
end

"""
Non-allocating version of `vectorizedReward`
"""
struct VecReward{T}
    dest::Vector{T}
end

VecReward() = VecReward(Vector{Float64}(undef, 1))

function (r::VecReward)(m, s, a, sp)
    return r.dest .= POMDPs.reward(m,s,a,sp)
end

function (r::VecReward)(m, s, a)
    return r.dest .= POMDPs.reward(m,s,a)
end



"""
    EvalPolicyGraph(m::POMDP,pg::PolicyGraph;tolerance::Float64=0.001,rewardfunction = POMDPs.reward)

    Evaluates a PolicyGraph using iteration.
    Returns a value matrix (with each column corresponding to a state and each row corresponding to a graph node)
    Default tolerance is 0.001
    Optionally pass a custom reward function to incorporate cost or other functions.
    This function must return a vector (splat any cost vectors).
    Modified from: https://jair.org/index.php/jair/article/view/11216
"""
function EvalPolicyGraph end

function EvalPolicyGraph(
    m::POMDP,
    pg::PolicyGraph;
    tolerance::Float64 = 0.001,
    rewardfunction = VecReward(),
    disc = discount(m)
    )

    #set based on the number of steps to relevant value
    # disc_io ? γ = discount(m) : γ = 0.99995
    # isa(disc,Vector) ? γ = diagm(disc) : γ = disc
    γ = disc

    a = rand(actions(m))
    s = rand(initialstate(m))
    sp = rand(initialstate(m))
    Nn = length(pg.nodes)
    Ns = length(states(m))
    rew_size = length(rewardfunction(m,s,a,sp))
    v = ones(Nn, Ns, rew_size)
    v_p = zeros(Nn, Ns, rew_size)
    diff_mat = Array{Float64, 3}(undef, Nn, Ns, rew_size)
    v_int = Vector{Float64}(undef, rew_size)
    v_tmp = copy(v_int)
    count = 0

    while norm(diff_mat .= v_p .- v, Inf) > tolerance
        count += 1
        v .= v_p
        for i in 1:length(pg.nodes)
            for (s_idx, s) in enumerate(ordered_states(m))
                if !isterminal(m, s)
                    a = pg.nodes[i]
                    v_int .= 0.
                    t_dist = transition(m, s, a)
                    for sp in support(t_dist)
                        sp_idx = stateindex(m,sp)
                        prob_t = pdf(t_dist,sp)
                        r = rewardfunction(m, s, a, sp)
                        @. v_int += prob_t*r
                        o_dist = observation(m, s, a, sp)
                        for o in support(o_dist)
                            prob_o = pdf(o_dist,o)
                            edge = get(pg.edges, (i,o), nothing)
                            if !isnothing(edge)
                                @inbounds copyto!(v_tmp, @view v[edge,sp_idx,:])
                                @. v_int += (v_tmp *= γ*prob_t*prob_o)
                                # v_int += (v_tmp = γ*prob_t*prob_o*v_tmp)
                            end
                        end
                    end
                    @inbounds copyto!(view(v_p, i,s_idx,:), v_int)
                end
            end
        end
    end
    return v_p
end


##Convenience Functions
"""
    GenandEvalPG(updater::Updater,pol::AlphaVectorPolicy,b0::DiscreteBelief;tolerance::Int64=0.001,rewardfunction=POMDPs.reward)

    Generates and evaluates (using iteration) an alpha-vector-based policy graph.
    Returns a value matrix (with each column corresponding to a state and each row corresponding to a graph node) and policy graph
    Default tolerance for termination of evaluation iteration is 0.001
    Optionally pass a custom reward function to incorporate cost or other functions.
    This function must return a vector (splat any cost vectors).
"""

function GenandEvalPG end
function GenandEvalPG(m::POMDP,updater::Updater,pol::AlphaVectorPolicy,b0::DiscreteBelief,depth::Int;eval_tolerance::Float64=0.001,rewardfunction=POMDPs.reward)
    pg = policy2fsc(m, updater, pol, b0, depth)
    values = EvalPolicyGraph(m,pg;tolerance=eval_tolerance,rewardfunction=rewardfunction)
    return values
end