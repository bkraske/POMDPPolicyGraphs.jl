# using POMDPPolicyGraphs
using POMDPs, POMDPTools, SARSOP
using RockSample, POMDPModels
using Statistics
using Test
using .POMDPPolicyGraphs

rs = RockSamplePOMDP(5, 7)
tiger = TigerPOMDP()
cb = BabyPOMDP()
tm = TMaze()
mh = MiniHallway()

function get_policy(m::POMDP; solver=SARSOPSolver(timeout=60, verbose=false))
    #Solve Problem
    pol = solve(solver, m)
    up = DiscreteUpdater(m)
    bel0 = initialize_belief(up, initialstate(m))
    return (m, up, pol, bel0)
end

function compare_pg_rollout(m::POMDP, up::Updater, pol::Policy, bel0::DiscreteBelief, pg_val;
    runs=1000)
    #Do MC Sims
    simlist = [Sim(m, pol, up, bel0; max_steps=1000) for i in 1:runs]
    mc_res_raw = run(simlist) do sim, hist
        return [:disc_rew => discounted_reward(hist)]
    end
    mc_res = mean(mc_res_raw[!, :disc_rew])
    mc_res_sem = 3 * std(mc_res_raw[!, :disc_rew]) / sqrt(runs)

    bel_val = BeliefValue(pg_val, bel0)
    #Compare and Report
    @show mc_res
    @show typeof(bel_val)
    @info "Difference is $(mc_res-bel_val[1]), 3 SEM is $mc_res_sem"
    @info "Passing: $((mc_res-bel_val[1])<mc_res_sem)"
    return (mc_res - bel_val[1]) < mc_res_sem
end

function pg_vs_mc(m::POMDP; solver=SARSOPSolver(; timeout=60, verbose=false))
    m_tuple = get_policy(m::POMDP; solver=solver)
    pg_res = GenandEvalPG(m_tuple...,25)
    return compare_pg_rollout(m_tuple..., pg_res)
end

@test pg_vs_mc(rs) #Intermittant Failure
@test pg_vs_mc(tiger)
@test pg_vs_mc(cb)
@test pg_vs_mc(mh)
@test pg_vs_mc(tm) #Consistently fails


