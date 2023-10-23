#Take Tabular from NativeSARSOP
struct EvalTabularPOMDP <: POMDP{Int,Int,Int} #From NativeSARSOP
    T::Vector{SparseMatrixCSC{Float64, Int64}} #T[a][sp,s]
    R::Array{Float64,3} # R[s,a,:]
    O::Vector{SparseMatrixCSC{Float64, Int64}} # O[a][sp, o]
    O2::Vector{SparseMatrixCSC{Float64, Int64}} # O[a][o, sp]
    isterminal::BitVector
    initialstate::SparseVector{Float64, Int}
    discount::Float64
end

function  EvalTabularPOMDP(pomdp::POMDP;rew_f=VecReward(),r_len=1)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)

    terminal = NativeSARSOP._vectorized_terminal(pomdp, S)
    T = transition_matrix_a_sp_s(pomdp)
    R = eval_tabular_rewards(pomdp, S, A, terminal, rew_f, r_len)
    O1 = POMDPTools.ModelTools.observation_matrix_a_sp_o(pomdp)
    O2 = map(sparse ∘ transpose,O1) ##from PBVI.jl, Tyler
    b0 = NativeSARSOP._vectorized_initialstate(pomdp, S)
    return EvalTabularPOMDP(T,R,O1,O2,terminal,b0,discount(pomdp))
end

##from HSVI4CGCP.jl, Tyler
function transition_matrix_a_sp_s(mdp::Union{MDP, POMDP})
    S = ordered_states(mdp)
    A = ordered_actions(mdp)

    ns = length(S)
    na = length(A)

    transmat_row_A = [Int64[] for _ in 1:na]
    transmat_col_A = [Int64[] for _ in 1:na]
    transmat_data_A = [Float64[] for _ in 1:na]

    for (si,s) in enumerate(S)
        for (ai,a) in enumerate(A)
            if isterminal(mdp, s) # if terminal, there is a probability of 1 of staying in that state
                push!(transmat_row_A[ai], si)
                push!(transmat_col_A[ai], si)
                push!(transmat_data_A[ai], 1.0)
            else
                td = transition(mdp, s, a)
                for (sp, p) in weighted_iterator(td)
                    if p > 0.0
                        spi = stateindex(mdp, sp)
                        push!(transmat_row_A[ai], spi)
                        push!(transmat_col_A[ai], si)
                        push!(transmat_data_A[ai], p)
                    end
                end
            end
        end
    end
    transmats_A_SP_S = [sparse(transmat_row_A[a], transmat_col_A[a], transmat_data_A[a], ns, ns) for a in 1:na]
    return transmats_A_SP_S
end

function eval_tabular_rewards(pomdp, S, A, terminal, rew_f, r_len)
    R =  Array{Float64}(undef, length(S), length(A), r_len)
    for (s_idx, s) ∈ enumerate(S)
        if terminal[s_idx]
            R[s_idx, :, :] .= 0.0
            continue
        end
        for (a_idx, a) ∈ enumerate(A)
            R[s_idx, a_idx, :] .= rew_f(pomdp, s, a)::Vector{Float64}
        end
    end
    R
end

function osp_tabular_observations(pomdp, S, A, O)
    _O = [Matrix{Float64}(undef, length(O), length(S)) for _ ∈ eachindex(A)]
    for i ∈ eachindex(_O)
        osp_fill_observations!(pomdp, _O[i], S, A[i], O)
    end
    _O
end

function osp_fill_observations!(pomdp, Oa, S, a, O)
    for (sp_idx, sp) ∈ enumerate(S)
        obs_dist = observation(pomdp, a, sp)
        for (o_idx, o) ∈ enumerate(O)
            Oa[o_idx, sp_idx] = pdf(obs_dist, o)
        end
    end
    Oa
end


POMDPTools.ordered_states(pomdp::EvalTabularPOMDP) = axes(pomdp.R, 1)
POMDPs.states(pomdp::EvalTabularPOMDP) = ordered_states(pomdp)
POMDPTools.ordered_actions(pomdp::EvalTabularPOMDP) = eachindex(pomdp.T)
POMDPs.actions(pomdp::EvalTabularPOMDP) = ordered_actions(pomdp)
POMDPTools.ordered_observations(pomdp::EvalTabularPOMDP) = axes(first(pomdp.O), 2)
POMDPs.observations(pomdp::EvalTabularPOMDP) = ordered_observations(pomdp)

POMDPs.discount(pomdp::EvalTabularPOMDP) = pomdp.discount

#Modified from NativeSARSOP
belief_reward(s_pomdp::EvalTabularPOMDP, b::SparseVector{Float64, Int64}, a::Int) = [dot(@view(s_pomdp.R[:,a,i]), b) for i in axes(s_pomdp.R,3)]

function corrector(pomdp::EvalTabularPOMDP, pred::AbstractVector, a, o::Int)
    return NativeSARSOP._sparse_col_mul(pred, pomdp.O[a], o)
end

function action_from_vec(pomdp::POMDP,pol::AlphaVectorPolicy,b::SparseVector{Float64, Int64})
    best_val = -Inf
    best_action = pol.action_map[1]
    for (i,α) in enumerate(pol.alphas)
        val = dot(b,α)
        if val > best_val
            best_action = pol.action_map[i]
            best_val = val
        end
    end
    return actionindex(pomdp,best_action)
end

function isterminalbelief(s_pomdp::EvalTabularPOMDP,b::SparseVector{Float64, Int64})
    return all(s_pomdp.isterminal[SparseArrays.nonzeroinds(b)])
end

function belief_value_exhaustive(pomdp::POMDP{S,A}, s_pomdp::EvalTabularPOMDP, updater::Updater, pol::Policy, b::SparseVector{Float64, Int64}, depth::Int, d::Int) where {S,A}
    a = action_from_vec(pomdp,pol, b)
    value = belief_reward(s_pomdp,b,a)
    if d<depth
        d+=1
        obs = s_pomdp.O[a]
        pred = s_pomdp.T[a]*b
        for o in axes(obs,2)
            bp = corrector(s_pomdp, pred, a, o)
            po = sum(bp)
            if po > 0. && !isterminalbelief(s_pomdp,bp)
                bp.nzval ./= po
                value += discount(s_pomdp)*po*belief_value_exhaustive(pomdp, s_pomdp, updater, pol, bp, depth, d)
            end    
        end
    end
    return value
end

#New Code
"""
    ExhaustiveEvaluator(m::POMDP,depth::Int)

    Instantiates an ExhaustiveEvaluator, which evaluates a POMDP policy by building 
    a policy tree which branches on all observations to some `depth` or until all beliefs 
    are terminal. Uses `DiscreteUpdater` by default.
"""
struct ExhaustiveEvaluator
    depth::Int
    updater::Updater
end

function ExhaustiveEvaluator(m::POMDP,depth::Int)
    return ExhaustiveEvaluator(depth,DiscreteUpdater(m))
end

"""
    Online value function for a policy on a POMDP given a belief.
    
    Calculates the value of a policy given some belief using exhaustive evaluation.
"""
struct EEValueFunction{M<:POMDP,R} <: Function
    m::M
    evaluator::ExhaustiveEvaluator
    pol::Policy
    rewardfunction::R
end

"""
    evaluate(evaluator::ExhaustiveEvaluator, m::POMDP{S,A}, pol::Policy; rewardfunction=VecReward())

    Returns an EEValueFunction, which calculates the value of a belief using a exhaustive evaluation.
    
    Optionally pass a custom reward function (which must return a vector) to incorporate cost or other functions.
"""
function POMDPTools.evaluate(evaluator::ExhaustiveEvaluator, m::POMDP{S,A}, pol::Policy; rewardfunction=VecReward()) where {S,A}
    @assert isa(evaluator.updater,DiscreteUpdater)
    return EEValueFunction(m,evaluator,pol,rewardfunction)
end

function (v::EEValueFunction)(b0)
    @assert b0.pomdp==v.m

    r_dim = length(v.rewardfunction(v.m,ordered_states(v.m)[1],ordered_actions(v.m)[1]))
    s_pomdp = EvalTabularPOMDP(v.m;rew_f=v.rewardfunction,r_len=r_dim)
    r = belief_value_exhaustive(v.m, s_pomdp, v.evaluator.updater, v.pol, sparse(b0.b), v.evaluator.depth, 1)
    return r
end

# function evaluate(evaluator::ExhaustiveEvaluator, pomdp::POMDP{S,A}, pol::Policy, b::DiscreteBelief; rewardfunction=VecReward()) where {S,A} #TYLER
#     updater = evaluator.updater
#     depth = evaluator.depth
#     d = 1
#     r_dim = length(rewardfunction(pomdp,ordered_states(pomdp)[1],ordered_actions(pomdp)[1]))
#     s_pomdp = EvalTabularPOMDP(pomdp;rew_f=rewardfunction,r_len=r_dim)
#     r = belief_value_exhaustive(pomdp, s_pomdp, updater, pol, sparse(b.b), depth, d)
#     return r
# end