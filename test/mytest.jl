using POMDPs, POMDPTools
using PBVI
using POMDPModels, RockSample

using ConstrainedPOMDPModels
using ConstrainedPOMDPs
using StaticArrays
using Statistics

using NativeSARSOP

rs = RockSamplePOMDP() #(5, 7)
tiger = TigerPOMDP()
cb = BabyPOMDP()
tm = TMaze()
mh = MiniHallway()

solver = PBVISolver(max_iter=40, verbose=true, witness_b=true) 
t_pol = solve(solver,tiger)
t_up = DiscreteUpdater(tiger)
t_b0 = initialize_belief(t_up,initialstate(tiger))
# t_pt = POMDPPolicyGraphs.policy_tree(tiger, t_up, t_pol[1], t_b0, 6)
# t_pg = policy2fsc(tiger, t_up, t_pol, t_b0, 6)
# t_pg_e = POMDPPolicyGraphs.GenandEvalPG(tiger, t_up, t_pol[1], t_b0, 6)


# tpgc = POMDPPolicyGraphs.CGCP_pg2(tiger, t_up, t_pol...)
# tpg = POMDPPolicyGraphs.CGCP2PG(tpgc)
# tpg1 = policy2fsc(tiger, tpg)
# tpt = POMDPPolicyGraphs.policy_tree(tiger, t_up, t_pol[1], t_b0, 7)
# tpg2 = policy2fsc(tiger, t_up, t_pol[1], t_b0, 5)

# val1 = POMDPPolicyGraphs.EvalPolicyGraph(tiger, tpg)
# val2 = POMDPPolicyGraphs.EvalPolicyGraph(tiger, tpg2)

# pg_val = BeliefValue(tiger, t_up, t_pol[1], t_b0, 5)


# #
# println("Instantiate Problem") 
# po_gw = ConstrainedPOMDPModels.GridWorldPOMDP(size=(5, 5),
#     terminate_from=Set(SVector{2,Int64}[[5, 5]]),
#     rewards=Dict(ConstrainedPOMDPModels.GWPos(5, 5) => 10.0),
#     tprob=1.0)
# c_gw = ConstrainedPOMDPs.Constrain(po_gw, [1.0])
# solver2 = PBVISolver(max_time=10.0)

# gw_up = DiscreteUpdater(po_gw)
# gw_b0 = initialize_belief(gw_up,initialstate(po_gw))
# gw_pol = solve(solver, po_gw)

# pg = policy2fsc(po_gw, gw_up, gw_pol[1], gw_b0, 5)
# tree = policy_tree(po_gw, gw_up, gw_pol[1], gw_b0, 5)
# PG_reward(m::ConstrainedPOMDPWrapper, s, a, sp) =  PG_reward(m, s, a)

# function PG_reward(m::ConstrainedPOMDPWrapper,s,a)
#     return [reward(m.m, s, a), ConstrainedPOMDPs.cost(m.m,s,a)...]
# end

# function POMDPs.reward(m::ConstrainedPOMDPModels.GridWorldPOMDP, s, a)
#     return reward(m.mdp, s, a) - [1]'*ConstrainedPOMDPs.cost(c_gw,s,a)
# end
# gw_pol = solve(solver, po_gw)

# pg_val = BeliefValue(c_gw, gw_up, gw_pol[1], gw_b0, 6)

@show value = recursive_evaluation(tiger, t_up, t_pol[1], VecReward(), t_b0, 6)

runs = 1000000
simlist = [Sim(tiger,t_pol[1], t_up,t_b0,max_steps=7) for i in 1:runs]
result = run_parallel(simlist) do sim, hist
    return [:disc_rew=>discounted_reward(hist)]
end
@show mu = mean(result.disc_rew)
@show sem = std(result.disc_rew)/sqrt(runs)
@show mu-3*sem < value[1] < mu+3*sem


#RS Testing
rs_pol = solve(SARSOPSolver(),rs)
rs_up = DiscreteUpdater(rs)
rs_b0 = initialize_belief(rs_up,initialstate(rs))
rs_value0 = BeliefValue(rs, rs_up, rs_pol, rs_b0, 30)
# rs_value = recursive_evaluation(rs, rs_up, sar_pol, VecReward(), rs_b0, 6)
my_vals = [BeliefValue(rs, rs_up, rs_pol, rs_b0, 100;replace=[a])[1] for a in ordered_actions(rs)]

#TEST PG ACCURACY HERE --> Compare to MC Sim and see if correct for diff first action
runs2 = 100000

function my_sim(pomdp,up,pol;replace=[],max_steps=typemax(Int))
    r_ave = []
    r_ave_u = []
    for _ in 1:runs2
        r_tot = 0
        r_tot_u = 0
        b = initialize_belief(up,initialstate(pomdp))
        s = rand(initialstate(pomdp))
        steps = 0
        d = 1.0
        while !isterminal(pomdp,s) && steps <= max_steps
            steps += 1
            if steps==1 && !isempty(replace)
                a = first(replace)
            else
                a = action(pol,b)
            end
            s,o,r = @gen(:sp,:o,:r)(pomdp,s,a)
            b = update(up,b,a,o)
            r_tot += d*r
            d*= discount(pomdp)
            r_tot_u += r
        end
        push!(r_ave,r_tot)
        push!(r_ave_u,r_tot)
    end
    return mean(r_ave), std(r_ave)/sqrt(runs2)
end

simlist2 = [Sim(rs,rs_pol, rs_up,rs_b0) for i in 1:runs2]
result2 = run_parallel(simlist2) do sim, hist
    return [:disc_rew=>undiscounted_reward(hist)]
end
@show mu2 = mean(result2.disc_rew)
@show sem2 = std(result2.disc_rew)/sqrt(runs2)
@show mu2-3*sem2 < rs_value0[1] < mu2+3*sem2