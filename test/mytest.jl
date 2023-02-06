using POMDPs, POMDPTools
using SARSOP
using POMDPModels, RockSample

rs = RockSamplePOMDP(5, 7)
tiger = TigerPOMDP()
cb = BabyPOMDP()
tm = TMaze()
mh = MiniHallway()

solver = SARSOPSolver() 
t_pol = solve(solver,tiger)
t_up = DiscreteUpdater(tiger)
t_b0 = initialize_belief(t_up,initialstate(tiger))
# t_pt = POMDPPolicyGraphs.policy_tree(tiger, t_up, t_pol, t_b0, 5)
# t_pg = policy2fsc(tiger, t_up, t_pol, t_b0, 5)
# t_pg_e = POMDPPolicyGraphs.GenandEvalPG(tiger, t_up, t_pol, t_b0, 5)
