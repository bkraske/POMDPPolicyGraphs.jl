# using POMDPPolicyGraphs
using POMDPs, POMDPTools, NativeSARSOP
using RockSample, POMDPModels
using Statistics
using Test
using ConstrainedPOMDPModels
# using .POMDPPolicyGraphs

rs = RockSamplePOMDP(5,7)
tiger = TigerPOMDP()
cb = BabyPOMDP()
tm = TMaze()
mh = MiniHallway()
gw = ConstrainedPOMDPModels.GridWorldPOMDP()

function get_policy(m::POMDP; solver=SARSOPSolver(;max_time=typemax(Float64)))
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

function pg_vs_mc(m::POMDP; solver=SARSOPSolver(;max_time=10.0),h=15,runs=5000)
    m_tuple = get_policy(m::POMDP; solver=solver)
    pg_res = gen_belief_value(m_tuple..., h)
    return compare_pg_rollout(m_tuple..., pg_res;h=500,runs=runs) #30000
end

function recur_vs_mc(m::POMDP; solver=SARSOPSolver(;max_time=10.0),h=15,runs=5000)
    m_tuple = get_policy(m::POMDP; solver=solver)
    pg_res = recursive_evaluation(m_tuple..., h)
    return compare_pg_rollout(m_tuple..., pg_res;h=h,runs=runs)
end

@testset "Policy Graph" begin
    testh = 40
    n_runs = 30000
    @test pg_vs_mc(tiger;h=testh,runs=n_runs)
    @test pg_vs_mc(cb;h=testh,runs=n_runs)
    @test pg_vs_mc(mh;h=testh,runs=n_runs)
    @test pg_vs_mc(tm;h=testh,runs=n_runs)
end

@testset "Recursive Evaluation" begin
    testh = 20
    n_runs = 30000
    @test recur_vs_mc(tiger;h=testh,runs=n_runs)
    @test recur_vs_mc(cb;h=testh,runs=n_runs)
    @test recur_vs_mc(mh;h=testh,runs=n_runs)
    @test recur_vs_mc(tm;h=testh,runs=n_runs)
end

@testset "RockSample Tests" begin
    testh = 45
    n_runs = 10000
    @test pg_vs_mc(rs;h=testh,runs=n_runs)
    @test recur_vs_mc(rs;h=testh,runs=n_runs)
end

@testset "RockSample sameness" begin
    solver=SARSOPSolver(;max_time=10.0)
    h=50
    runs=30000
    m_tuple = get_policy(rs; solver=solver)
    pg_res = gen_belief_value(m_tuple..., h)
    @info pg_res[1]
    recur_res = recursive_evaluation(m_tuple..., h)[1]
    @info recur_res
    @show pg_res[1]-recur_res
    @test isapprox(pg_res[1],recur_res;atol=0.0001)
    old_pg = gen_belief_value(m_tuple..., h,old_eval=true)
    @show old_pg[1]
    @show old_pg[1]-pg_res[1]
    @test isapprox(old_pg[1],recur_res;atol=0.0001)
    # @test compare_pg_rollout(m_tuple..., pg_res;h=500,runs=runs)
end

# @testset "GridWorldPOMDP" begin
#     h = 5
#     n_runs = 10000
#     @test pg_vs_mc(gw;h=testh,runs=n_runs)
#     @test recur_vs_mc(gw;h=testh,runs=n_runs)
# end