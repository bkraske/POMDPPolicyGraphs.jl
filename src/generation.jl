##Structs
struct PolicyGraph{N,E} <: Policy
    nodes::N
    edges::E
    node1::Int64
    beliefs::Vector{SparseVector{Float64, Int64}}
    node_depth::Vector{Int}
end

##TO DO: Implement function for checking if belief is terminal

##Recursive Tree Method
"""
    gen_polgraph(m::POMDP{S,A}, s_pomdp::EvalTabularPOMDP, updater::Updater, pol::Policy, b0::DiscreteBelief, depth::Int; replace::Vector=A[],return_beliefs::Bool=false)
    gen_polgraph(m::POMDP{S,A}, updater::Updater, pol::Policy, b0::DiscreteBelief, depth::Int; replace::Vector=A[], store_beliefs::Bool=false)

    Generates a policy graph up to specified `depth`.
    Optionally replace the first action in the Policy Graph with an alternative action, e.g. `replace=[:up]`
    Optionally returns beliefs used to label nodes in PolicyGraph in the `PolicyGraph` struct.
"""
function gen_polgraph end

function gen_polgraph(m::POMDP, s_pomdp::EvalTabularPOMDP, updater::Updater, pol::Policy, b0::SparseVector, depth::Int, action_list, edge_list, b_list, d, j_old, a_old, oo, oa, replace, depth_list)
    if d < depth
        d+=1
        obs = s_pomdp.O[a_old]
        pred = s_pomdp.T[a_old]*b0
        for o in axes(obs,2)
            bp = corrector(s_pomdp, pred, a_old, o)
            po = sum(bp)
            if po > 0. && !isterminalbelief(s_pomdp,bp)
                bp.nzval ./= po
                if replace
                    bp_idx = findall(x->x==bp, b_list[2:end]) .+ 1
                else
                    bp_idx = findall(x->x==bp, b_list)
                end
                if !isempty(bp_idx) #bp ∈ b_list
                    push!(edge_list, (j_old, oo[o]) => bp_idx[1])
                else
                    a = action_from_vec(m, pol, bp)
                    push!(action_list, oa[a])
                    push!(b_list,bp)
                    push!(depth_list,d)
                    # @show Vector.(b_list)
                    j = copy(length(action_list))
                    push!(edge_list, (j_old, oo[o]) => j)

                    gen_polgraph(m,s_pomdp,updater,pol,bp,depth,action_list,edge_list,b_list,d,j,a,oo,oa,replace,depth_list)
                end
            end    
        end
    end
end

function gen_polgraph(m::POMDP{S,A}, s_pomdp::EvalTabularPOMDP, updater::Updater, pol::Policy, b0::DiscreteBelief, depth::Int; replace::Vector=A[],store_beliefs::Bool=false) where {S,A}
    edge_list = Dict{Tuple{Int64,obstype(pol.pomdp)},Int64}()
    action_list = A[]
    b_list = SparseVector{Float64, Int64}[]
    depth_list = Int[]
    d = 1
    a=if !isempty(replace)
        replace[1]
    else
        action(pol, b0)
    end::A
    push!(action_list, a)
    push!(b_list,sparse(b0.b))
    push!(depth_list,d)
    j = copy(length(action_list))

    oo = ordered_observations(m)
    oa = ordered_actions(m)

    gen_polgraph(m, s_pomdp, updater, pol, sparse(b0.b), depth, action_list, edge_list, b_list, d, j, actionindex(m,a), oo, oa, !isempty(replace),depth_list)
    if !store_beliefs
        return PolicyGraph(action_list, edge_list, 1, SparseVector{Float64, Int64}[],depth_list)
    else
        return PolicyGraph(action_list, edge_list, 1, b_list, depth_list)
    end
end

function gen_polgraph(m::POMDP{S,A}, updater::Updater, pol::Policy, b0::DiscreteBelief, depth::Int; replace::Vector=A[], store_bels::Bool=false) where {S,A}
    s_pomdp = EvalTabularPOMDP(m)
    return gen_polgraph(m, s_pomdp, updater, pol, b0, depth; replace=replace, store_beliefs=store_bels)
end