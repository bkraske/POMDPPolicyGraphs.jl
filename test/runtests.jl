# using POMDPPolicyGraphs
using POMDPs, POMDPTools, NativeSARSOP
using RockSample, POMDPModels
using Statistics
using Test
# using .POMDPPolicyGraphs

rs = RockSamplePOMDP(5,7)
tiger = TigerPOMDP()
cb = BabyPOMDP()
tm = TMaze()
mh = MiniHallway()

function get_policy(m::POMDP; solver=SARSOPSolver())
    #Solve Problem
    pol = solve(solver, m)
    up = DiscreteUpdater(m)
    bel0 = initialize_belief(up, initialstate(m))
    return (m, up, pol, bel0)
end

function compare_pg_rollout(m::POMDP, up::Updater, pol::Policy, bel0::DiscreteBelief, pg_val;
    runs=5000,h=15)
    @info m
    #Do MC Sims
    simlist = [Sim(m, pol, up, bel0; max_steps=h) for _ in 1:runs]
    mc_res_raw = run(simlist) do sim, hist
        return [:disc_rew => discounted_reward(hist)]
    end
    mc_res = mean(mc_res_raw[!, :disc_rew])
    mc_res_sem = 3 * std(mc_res_raw[!, :disc_rew]) / sqrt(runs)

    bel_val = pg_val[1]
    #Compare and Report
    @show mc_res
    @show bel_val[1]
    is_pass = (abs(mc_res-bel_val)<mc_res_sem)
    @info "Difference is $(mc_res-bel_val), 3 SEM is $mc_res_sem"
    @info "Passing: $is_pass"
    return is_pass
end

function pg_vs_mc(m::POMDP; solver=SARSOPSolver(),h=15)
    m_tuple = get_policy(m::POMDP; solver=solver)
    pg_res = gen_belief_value(m_tuple..., h)
    return compare_pg_rollout(m_tuple..., pg_res;h=500)
end

function recur_vs_mc(m::POMDP; solver=SARSOPSolver(),h=15)
    m_tuple = get_policy(m::POMDP; solver=solver)
    pg_res = recursive_evaluation(m_tuple..., h)
    return compare_pg_rollout(m_tuple..., pg_res;h=h)
end

@testset "Policy Graph" begin
    testh = 20
    @test pg_vs_mc(rs;h=testh) #Intermittant Failure
    @test pg_vs_mc(tiger;h=testh)
    @test pg_vs_mc(cb;h=testh)
    @test pg_vs_mc(mh;h=testh)
    @test pg_vs_mc(tm;h=testh) #Consistently fails
end

@testset "Recursive Evaluation" begin
    testh = 7
    @test recur_vs_mc(rs;h=testh)
    @test recur_vs_mc(tiger;h=testh)
    @test recur_vs_mc(cb;h=testh)
    @test recur_vs_mc(mh;h=testh)
    @test recur_vs_mc(tm;h=testh)
end

rs_tuple = get_policy(rs)
rst1 = POMDPPolicyGraphs.gpg2pg(policy_tree(rs_tuple..., 15))
rst2 = POMDPPolicyGraphs.recursive_tree(rs_tuple..., 15)
rst3 = POMDPPolicyGraphs.sparse_recursive_tree(rs_tuple..., 15)

t_tuple = get_policy(tiger)
tt1 = POMDPPolicyGraphs.gpg2pg(policy_tree(t_tuple..., 15))
tt2 = POMDPPolicyGraphs.recursive_tree(t_tuple..., 15)
tt3 = POMDPPolicyGraphs.sparse_recursive_tree(t_tuple..., 15)
tt4 = policy2fsc(t_tuple..., 15)