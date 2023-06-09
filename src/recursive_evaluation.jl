#Take Tabular from ConstrainedSARSOP
struct EvalTabularPOMDP <: POMDP{Int,Int,Int} #From ConstrainedSARSOP
    T::Vector{SparseMatrixCSC{Float64, Int64}}
    R::Array{Float64,3} # R[s,a]
    O::Vector{SparseMatrixCSC{Float64, Int64}} # O[a][sp, o]
    isterminal::BitVector
    initialstate::SparseVector{Float64, Int}
    discount::Float64
end

function  EvalTabularPOMDP(pomdp::POMDP;rew_f=VecReward(),r_len=1)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)

    T = _tabular_transitions(pomdp, S, A)
    R = _tabular_rewards(pomdp, S, A, rew_f, r_len)
    O = _tabular_observations(pomdp, S, A, O)
    term = _vectorized_terminal(pomdp, S)
    b0 = _vectorized_initialstate(pomdp, S)
    return EvalTabularPOMDP(T,R,O,term,b0,discount(pomdp))
end

function _tabular_transitions(pomdp, S, A)
    T = [Matrix{Float64}(undef, length(S), length(S)) for _ ∈ eachindex(A)]
    for i ∈ eachindex(T)
        _fill_transitions!(pomdp, T[i], S, A[i])
    end
    T
end

function _fill_transitions!(pomdp, T, S, a)
    for (s_idx, s) ∈ enumerate(S)
        Tsa = transition(pomdp, s, a)
        for (sp_idx, sp) ∈ enumerate(S)
            T[sp_idx, s_idx] = pdf(Tsa, sp)
        end
    end
    T
end

function _tabular_rewards(pomdp, S, A, rew_f, r_len)
    R = Array{Float64}(undef, length(S), length(A), r_len)
    for (s_idx, s) ∈ enumerate(S)
        for (a_idx, a) ∈ enumerate(A)
            R[s_idx, a_idx,:] = rew_f(pomdp, s, a)
        end
    end
    R
end

function _tabular_observations(pomdp, S, A, O)
    _O = [Matrix{Float64}(undef, length(S), length(O)) for _ ∈ eachindex(A)]
    for i ∈ eachindex(_O)
        _fill_observations!(pomdp, _O[i], S, A[i], O)
    end
    _O
end

function _fill_observations!(pomdp, Oa, S, a, O)
    for (sp_idx, sp) ∈ enumerate(S)
        obs_dist = observation(pomdp, a, sp)
        for (o_idx, o) ∈ enumerate(O)
            Oa[sp_idx, o_idx] = pdf(obs_dist, o)
        end
    end
    Oa
end

function _tabular_costs(pomdp, S, A)
    n_c = ConstrainedPOMDPs.constraint_size(pomdp)
    C = Array{Float64, 3}(undef, length(S), length(A), n_c)
    for (s_idx,s) ∈ enumerate(S)
        for (a_idx,a) ∈ enumerate(A)
            C[s_idx, a_idx, :] .= cost(pomdp, s, a)
        end
    end
    C
end

function _vectorized_terminal(pomdp, S)
    term = BitVector(undef, length(S))
    @inbounds for i ∈ eachindex(term,S)
        term[i] = isterminal(pomdp, S[i])
    end
    return term
end

function _vectorized_initialstate(pomdp, S)
    b0 = initialstate(pomdp)
    b0_vec = Vector{Float64}(undef, length(S))
    @inbounds for i ∈ eachindex(S, b0_vec)
        b0_vec[i] = pdf(b0, S[i])
    end
    return sparse(b0_vec)
end

POMDPTools.ordered_states(pomdp::EvalTabularPOMDP) = axes(pomdp.R, 1)
POMDPs.states(pomdp::EvalTabularPOMDP) = ordered_states(pomdp)
POMDPTools.ordered_actions(pomdp::EvalTabularPOMDP) = eachindex(pomdp.T)
POMDPs.actions(pomdp::EvalTabularPOMDP) = ordered_actions(pomdp)
POMDPTools.ordered_observations(pomdp::EvalTabularPOMDP) = axes(first(pomdp.O), 2)
POMDPs.observations(pomdp::EvalTabularPOMDP) = ordered_observations(pomdp)

POMDPs.discount(pomdp::EvalTabularPOMDP) = pomdp.discount

belief_reward(s_pomdp, b, a) = [dot(@view(s_pomdp.R[:,a,i]), b) for i in axes(s_pomdp.R,3)]


#FROM NativeSARSOP
function _sparse_col_mul(x::SparseVector{T}, A::SparseMatrixCSC{T}, col::Int) where T
    n = length(x)
    xnzind = SparseArrays.nonzeroinds(x)
    xnzval = SparseArrays.nonzeros(x)

    Anzr = nzrange(A, col)
    Anzval = @view nonzeros(A)[Anzr]
    Anzind = @view rowvals(A)[Anzr]

    mx = length(xnzind)
    mA = length(Anzr)

    cap = min(mx,mA)
    rind = zeros(Int, cap)
    rval = zeros(T, cap)
    ir = 0
    ix = 1
    iy = 1

    ir = SparseArrays._binarymap_mode_0!(*, mx, mA, xnzind, xnzval, Anzind, Anzval, rind, rval)
    resize!(rind, ir)
    resize!(rval, ir)
    return SparseVector(n, rind, rval)
end

function corrector(pomdp::EvalTabularPOMDP, pred::AbstractVector, a, o::Int)
    return _sparse_col_mul(pred, pomdp.O[a], o)
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

#New Code

"""
    recursive_evaluation(pomdp::POMDP, updater::Updater, pol::Policy, rew_f, b::DiscreteBelief, depth::Int)

Calculates the value of a policy recursively to a specified depth, calculating reward according to 'rew_f', the reward function passed.

"""
function recursive_evaluation end


function recursive_evaluation(pomdp::POMDP{S,A}, updater::Updater, pol::Policy, rew_f, b::DiscreteBelief, depth::Int) where {S,A} #TYLER
    d = 1
    s_pomdp = EvalTabularPOMDP(pomdp)
    r_dim = length(rew_f(pomdp,ordered_states(pomdp)[1],ordered_actions(pomdp)[1],ordered_states(pomdp)[1]))
    r = recursive_evaluation(pomdp, s_pomdp, updater, pol, rew_f, r_dim, sparse(b.b), depth, d)
    return r
end

function recursive_evaluation(pomdp::POMDP{S,A}, s_pomdp::EvalTabularPOMDP, updater::Updater, pol::Policy, rew_f, r_dim::Int64, b::SparseVector{Float64, Int64}, depth::Int, d::Int) where {S,A}
    a = action_from_vec(pomdp,pol, b)
    value = belief_reward(s_pomdp,b,a)
    if d<depth
        obs = s_pomdp.O[a]
        pred = s_pomdp.T[a]*b
        for o in axes(obs,2)
            bp = corrector(s_pomdp, pred, a, o)
            po = sum(bp)
            if po > 0.
                bp.nzval ./= po
                value += discount(s_pomdp)*po*recursive_evaluation(pomdp, s_pomdp, updater, pol, rew_f, r_dim, bp, depth, d+1)
            end    
        end
    end
    return value
end