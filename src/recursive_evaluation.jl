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
    O = ordered_observations(pomdp)

    terminal = NativeSARSOP._vectorized_terminal(pomdp, S)
    T = NativeSARSOP._tabular_transitions(pomdp, S, A, terminal)
    R = eval_tabular_rewards(pomdp, S, A, terminal, rew_f, r_len)
    O1 = NativeSARSOP._tabular_observations(pomdp, S, A, O)
    O2 = osp_tabular_observations(pomdp, S, A, O)
    b0 = NativeSARSOP._vectorized_initialstate(pomdp, S)
    return EvalTabularPOMDP(T,R,O1,O2,terminal,b0,discount(pomdp))
end

function eval_tabular_rewards(pomdp, S, A, terminal, rew_f, r_len)
    R =  Array{Float64}(undef, length(S), length(A), r_len)
    for (s_idx, s) ∈ enumerate(S)
        if terminal[s_idx]
            R[s_idx, :, :] .= 0.0
            continue
        end
        for (a_idx, a) ∈ enumerate(A)
            R[s_idx, a_idx, :] .= rew_f(pomdp, s, a)
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

#New Code

"""
    belief_value_recursive(pomdp::POMDP{S,A}, updater::Updater, pol::Policy, b::DiscreteBelief, depth::Int;rewardfunction=VecReward(),replace::Vector=A[])

    Calculates the value of a policy recursively to a specified depth, calculating reward according to `rew_f``, the reward function passed.
    Optionally replace the first action in the Policy Graph with an alternative action, e.g. `replace=[:up]`

"""
function belief_value_recursive end

function belief_value_recursive(pomdp::POMDP{S,A}, updater::Updater, pol::Policy, b::DiscreteBelief, depth::Int;rewardfunction=VecReward(),replace::Vector=A[]) where {S,A} #TYLER
    d = 1
    r_dim = length(rewardfunction(pomdp,ordered_states(pomdp)[1],ordered_actions(pomdp)[1]))
    s_pomdp = EvalTabularPOMDP(pomdp;rew_f=rewardfunction,r_len=r_dim)
    r = belief_value_recursive(pomdp, s_pomdp, updater, pol, sparse(b.b), depth, d, replace)
    return r
end

function belief_value_recursive(pomdp::POMDP{S,A}, s_pomdp::EvalTabularPOMDP, updater::Updater, pol::Policy, b::SparseVector{Float64, Int64}, depth::Int, d::Int, replace::Vector{A}) where {S,A}
    a=if d==1 && !isempty(replace)
        replace[1]
    else
        action_from_vec(pomdp,pol, b)
    end
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
                value += discount(s_pomdp)*po*belief_value_recursive(pomdp, s_pomdp, updater, pol, bp, depth, d, replace)
            end    
        end
    end
    return value
end