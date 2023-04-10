using POMDPs, POMDPTools
using PBVI
using POMDPModels, RockSample

using ConstrainedPOMDPModels
using ConstrainedPOMDPs
using StaticArrays

rs = RockSamplePOMDP(5, 7)
tiger = TigerPOMDP()
cb = BabyPOMDP()
tm = TMaze()
mh = MiniHallway()

solver = PBVISolver(max_iter=10, verbose=true, witness_b=true) 
t_pol = solve(solver,tiger)
t_up = DiscreteUpdater(tiger)
t_b0 = initialize_belief(t_up,initialstate(tiger))
# t_pt = POMDPPolicyGraphs.policy_tree(tiger, t_up, t_pol, t_b0, 5)
# t_pg = policy2fsc(tiger, t_up, t_pol, t_b0, 5)
# t_pg_e = POMDPPolicyGraphs.GenandEvalPG(tiger, t_up, t_pol, t_b0, 5)


tpgc = POMDPPolicyGraphs.CGCP_pg2(tiger, t_up, t_pol...)
tpg = POMDPPolicyGraphs.CGCP2PG(tpgc)
tpg1 = policy2fsc(tiger, tpg)
tpt = POMDPPolicyGraphs.policy_tree(tiger, t_up, t_pol[1], t_b0, 7)
tpg2 = policy2fsc(tiger, t_up, t_pol[1], t_b0, 5)

val1 = POMDPPolicyGraphs.EvalPolicyGraph(tiger, tpg)
val2 = POMDPPolicyGraphs.EvalPolicyGraph(tiger, tpg2)

pg_val = BeliefValue(tiger, t_up, t_pol[1], t_b0, 5)


#
println("Instantiate Problem") 
po_gw = ConstrainedPOMDPModels.GridWorldPOMDP(size=(5, 5),
    terminate_from=Set(SVector{2,Int64}[[5, 5]]),
    rewards=Dict(ConstrainedPOMDPModels.GWPos(5, 5) => 10.0),
    tprob=1.0)
c_gw = ConstrainedPOMDPs.Constrain(po_gw, [1.0])
solver2 = PBVISolver(max_iter=5)
gw_pol = solve(solver, po_gw)
gw_up = DiscreteUpdater(po_gw)
gw_b0 = initialize_belief(gw_up,initialstate(po_gw))

# pg = policy2fsc(po_gw, gw_up, gw_pol[1], gw_b0, 5)
# PG_reward(m::ConstrainedPOMDPWrapper, s, a, sp) =  PG_reward(m, s, a)

# function PG_reward(m::ConstrainedPOMDPWrapper,s,a)
#     return [reward(m.m, s, a), ConstrainedPOMDPs.cost(m.m,s,a)...]
# end
pg_val = BeliefValue(c_gw, gw_up, gw_pol[1], gw_b0, 6)