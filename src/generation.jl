##Structs
struct PolicyGraph{N,E} <: Policy
    nodes::N
    edges::E
    node1::Int64
end

##Recursive Tree Method
"""
    sparse_recursive_tree(m::POMDP{S,A}, s_pomdp::EvalTabularPOMDP, updater::Updater, pol::Policy, b0::DiscreteBelief, depth::Int; replace::Vector=A[],return_bels=false)

    Generates a policy graph up to specified `depth`.
    Optionally replace the first action in the Policy Graph with an alternative action, e.g. `replace=[:up]`
    Optionally returns beliefs used to label nodes in PolicyGraph.
"""

function sparse_recursive_tree end

function sparse_recursive_tree(m::POMDP, s_pomdp::EvalTabularPOMDP, updater::Updater, pol::Policy, b0::SparseVector, depth::Int, action_list, edge_list, b_list, d, j_old, a_old, oo, oa)
    if d < depth
        d+=1
        obs = s_pomdp.O[a_old]
        pred = s_pomdp.T[a_old]*b0
        for o in axes(obs,2)
            bp = corrector(s_pomdp, pred, a_old, o)
            po = sum(bp)
            # @show po > 0.
            if po > 0.
                bp.nzval ./= po
                # @show Vector(bp) ∈ Vector.(b_list)
                # @show Vector.(b_list)
                # @show bp
                # @show b_list
                bp_idx = findall(x->x==bp, b_list)
                if !isempty(bp_idx) #bp ∈ b_list
                    push!(edge_list, (j_old, oo[o]) => bp_idx[1])
                else
                    a = action_from_vec(m, pol, bp)
                    push!(action_list, oa[a])
                    push!(b_list,bp)
                    # @show Vector.(b_list)
                    j = copy(length(action_list))
                    push!(edge_list, (j_old, oo[o]) => j)

                    sparse_recursive_tree(m,s_pomdp,updater,pol,bp,depth,action_list,edge_list,b_list,d,j,a,oo,oa)
                end
            end    
        end
    end
end

function sparse_recursive_tree(m::POMDP{S,A}, s_pomdp::EvalTabularPOMDP, updater::Updater, pol::Policy, b0::DiscreteBelief, depth::Int; replace::Vector=A[],return_bels=false) where {S,A}
    edge_list = Dict{Tuple{Int64,obstype(pol.pomdp)},Int64}()
    action_list = A[]
    b_list = SparseVector{Float64, Int64}[]
    # s_pomdp = EvalTabularPOMDP(m)
    d = 1
    a=if !isempty(replace)
        replace[1]
    else
        action(pol, b0)
    end::A
    push!(action_list, a)
    push!(b_list,sparse(b0.b))
    j = copy(length(action_list))

    oo = ordered_observations(m)
    oa = ordered_actions(m)
    sparse_recursive_tree(m, s_pomdp, updater, pol, sparse(b0.b), depth, action_list, edge_list, b_list, d, j, actionindex(m,a), oo, oa)
    if !return_bels
        return PolicyGraph(action_list, edge_list, 1)
    else
        return PolicyGraph(action_list, edge_list, 1), b_list
    end
end