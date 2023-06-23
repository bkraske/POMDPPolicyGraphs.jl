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



"""
    eval_pg(m::POMDP,pg::PolicyGraph;tolerance::Float64=0.001,rewardfunction = POMDPs.reward)

    Evaluates a PolicyGraph using iteration.
    Returns a value matrix (with each column corresponding to a state and each row corresponding to a graph node)
    Default tolerance is 0.001
    Optionally pass a custom reward function to incorporate cost or other functions.
    This function must return a vector (splat any cost vectors).
    Modified from: https://jair.org/index.php/jair/article/view/11216
"""
function eval_pg end

function eval_pg(
    m::POMDP{S,A},
    pg::PolicyGraph;
    tolerance::Float64=0.001,
    rewardfunction=VecReward(),
    disc=discount(m)
) where {S,A}

    #set based on the number of steps to relevant value
    # disc_io ? γ = discount(m) : γ = 0.99995
    # isa(disc,Vector) ? γ = diagm(disc) : γ = disc
    γ = disc

    a = rand(actions(m))
    s = rand(initialstate(m))
    sp = rand(initialstate(m))
    Nn = length(pg.nodes)
    Ns = length(states(m))
    rew_size = length(rewardfunction(m, s, a, sp))
    v = ones(Nn, Ns, rew_size)
    v_p = zeros(Nn, Ns, rew_size)
    diff_mat = Array{Float64,3}(undef, Nn, Ns, rew_size)
    v_int = Vector{Float64}(undef, rew_size)
    v_tmp = copy(v_int)
    count = 0

    os = ordered_states(m)
    while norm(diff_mat .= v_p .- v, Inf) > tolerance
        count += 1
        v .= v_p
        for i in 1:length(pg.nodes)
            for (s_idx, s) in enumerate(os)
                if !isterminal(m, s)
                    a = pg.nodes[i]::A
                    v_int .= 0.0
                    t_dist = transition(m, s, a)
                    for sp in support(t_dist)
                        sp_idx = stateindex(m, sp)::Int
                        prob_t = pdf(t_dist, sp)
                        if prob_t > 0.0
                            r = rewardfunction(m, s, a, sp)
                            @. v_int += prob_t * r
                            o_dist = observation(m, s, a, sp)
                            for o in support(o_dist)
                                prob_o = pdf(o_dist, o)
                                node = get(pg.edges, (i, o), nothing)
                                if !isnothing(node)
                                    @inbounds copyto!(v_tmp, @view v[node::Int, sp_idx, :])
                                    @. v_int += (v_tmp *= γ * prob_t * prob_o)
                                    # v_int += (v_tmp = γ*prob_t*prob_o*v_tmp)
                                end
                            end
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
    pg::PolicyGraph;
    tolerance::Float64=0.001,
    rewardfunction=VecReward(),
    disc=discount(m)
) where {S,A}

    #set based on the number of steps to relevant value
    # disc_io ? γ = discount(m) : γ = 0.99995
    # isa(disc,Vector) ? γ = diagm(disc) : γ = disc
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
                        # o_dist = @view s_m.O2[a_idx][:,sp_idx]
                        # for o in SparseArrays.nonzeroinds(o_dist)
                        #     prob_o = o_dist[o]
                        #     node = s_edges[i][o]
                        #     if node != 0
                        #         @inbounds copyto!(v_tmp, @view v[node::Int, sp_idx, :])
                        #         @. v_int += (v_tmp *= γ * prob_t * prob_o)
                        #     end
                        # end
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
    rewardfunction=VecReward(),
    disc=discount(m)
) where {S,A}

    #set based on the number of steps to relevant value
    # disc_io ? γ = discount(m) : γ = 0.99995
    # isa(disc,Vector) ? γ = diagm(disc) : γ = disc
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
                        # o_dist = @view s_m.O2[a_idx][:,sp_idx]
                        # for o in SparseArrays.nonzeroinds(o_dist)
                        #     prob_o = o_dist[o]
                        #     node = s_edges[i][o]
                        #     if node != 0
                        #         @inbounds copyto!(v_tmp, @view v[node::Int, sp_idx, :])
                        #         @. v_int += (v_tmp *= γ * prob_t * prob_o)
                        #     end
                        # end
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
    gen_eval_pg(m::POMDP, updater::Updater, pol::AlphaVectorPolicy, b0::DiscreteBelief, depth::Int, eval_tolerance::Float64=0.001, rewardfunction=VecReward())

    Generates and evaluates (using iteration) an alpha-vector-based policy graph.
    Returns a value matrix (with each column corresponding to a state and each row corresponding to a graph node) and policy graph
    Default tolerance for termination of evaluation iteration is 0.001
    Optionally pass a custom reward function to incorporate cost or other functions.
    This function must return a vector (splat any cost vectors).
"""

function gen_eval_pg end
function gen_eval_pg(m::POMDP, updater::Updater, pol::AlphaVectorPolicy, 
            b0::DiscreteBelief, depth::Int; 
            eval_tolerance::Float64=0.001, rewardfunction=VecReward())
    # @show rewardfunction
    pg = policy2fsc(m, updater, pol, b0, depth)
    values = eval_pg(m, pg; tolerance=eval_tolerance, rewardfunction=rewardfunction)
    return values
end


"""
    get_belief_value(pg,result::Array, b::DiscreteBelief)

    Takes the state and node matrix from an evaluated policy graph and a Discrete Belief.
    Returns value of initial belief using the state values of the first node in the graph.
"""
function get_belief_value end

function get_belief_value(pg,result::Array, b::DiscreteBelief)
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
    gen_belief_value(m::POMDP, updater::Updater, pol::AlphaVectorPolicy, b0::DiscreteBelief, depth::Int; replace=[], eval_tolerance::Float64=0.001, rewardfunction=VecReward())

    Returns value of initial belief using the state values of the first node in the graph.
"""
function gen_belief_value end

function gen_belief_value(m::POMDP, updater::Updater, pol::AlphaVectorPolicy, 
            b0::DiscreteBelief, depth::Int; replace=[],
            eval_tolerance::Float64=0.001, rewardfunction=VecReward(), beliefbased=true)
    # @show rewardfunction
    # println("Generate PG")
    a = rand(actions(m))
    s = rand(initialstate(m))
    rew_size = length(rewardfunction(m, s, a))

    s_m = EvalTabularPOMDP(m;rew_f=rewardfunction,r_len = rew_size)

    if beliefbased == false
        # pg = policy2fsc(m, updater, pol, b0, depth;replace=replace)
        pg = sparse_recursive_tree(m, s_m, updater, pol, b0, depth;replace=replace)
        values = sparse_eval_pg(m, s_m, pg; tolerance=eval_tolerance, rewardfunction=rewardfunction)
    else
        pg,bels = sparse_recursive_tree(m, s_m, updater, pol, b0, depth;replace=replace,return_bels=true)
        values = sparse_eval_pg(m, s_m, pg, bels; tolerance=eval_tolerance, rewardfunction=rewardfunction)
    end
    # println("Evaluate PG")
    
    # values = eval_pg(m, pg; tolerance=eval_tolerance, rewardfunction=rewardfunction)
   
    i = pg.node1
    first_node = values[i, :, :]
    if length(support(b0)) == size(first_node)[1]
        return b0.b' * first_node
    else
        throw("Belief and result columns are different
              sizes: $(length(support(b))), $(size(first_node)[1])")
    end
end

function edge_dict_to_array(m,pg)
    # @show length(ordered_observations(m))
    E = [sparse(Int.(zeros(length(ordered_observations(m))))) for _ in eachindex(pg.nodes)]
    for node in eachindex(pg.nodes)
        # @show node
        for (i,o) in enumerate(ordered_observations(m))
            new_node = get(pg.edges, (node, o), nothing)
            # @show new_node
            if !isnothing(new_node)
                E[node][i] = new_node
            end
        end
    end
    return E
end