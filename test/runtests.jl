# include("restore_unregistered.jl")
# using POMDPPolicyGraphs
using POMDPs, POMDPTools, NativeSARSOP
using RockSample, POMDPModels
using Statistics
using Test
# using ConstrainedPOMDPs
# using ConstrainedPOMDPModels

rs = RockSamplePOMDP(5,7)
tiger = TigerPOMDP()
cb = BabyPOMDP()
tm = TMaze()
mh = MiniHallway()
# gw = ConstrainedPOMDPModels.GridWorldPOMDP()

function get_policy(m::POMDP; solver=SARSOPSolver(;max_time=10.0))
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

function compare_pg_rollout_vec(m::POMDP, up::Updater, pol::Policy, bel0::DiscreteBelief, pg_val, rew_f, r_len;
    runs=5000,h=15)
    @info m
    @show h
    #Do MC Sims
    results = []
    for _ in 1:runs
        s = rand(bel0)
        b = bel0
        r_total = zeros(r_len)
        d = 1.0
        count = 0
        while !isterminal(m, s) && count < h
            count += 1
            a = action(pol, b)
            r_total .+= d*rew_f(m,s,a)
            s, o = @gen(:sp,:o)(m, s, a)
            b = update(up,b,a,o)
            d *= discount(m)
        end
        push!(results,r_total)
    end
    mc_res = mean(results)
    mc_res_sem = 3 * std(results) / sqrt(runs)

    bel_val = pg_val
    #Compare and Report
    @info pg_res[1]==pg_res[2]
    @show mc_res
    @show bel_val
    # @show typeof(mc_res)
    # @show typeof(bel_val')
    is_pass = (abs.(mc_res-bel_val')<mc_res_sem)
    @info "Difference is $(mc_res-bel_val'), 3 SEM is $mc_res_sem"
    @info "Passing: $is_pass"
    return is_pass
end

function compare_r_rollout_vec(m::POMDP, up::Updater, pol::Policy, bel0::DiscreteBelief, pg_val, rew_f, r_len;
    runs=5000,h=15)
    @info m
    @show h
    #Do MC Sims
    results = []
    for _ in 1:runs
        s = rand(bel0)
        b = bel0
        r_total = zeros(r_len)
        d = 1.0
        count = 0
        while !isterminal(m, s) && count < h
            count += 1
            a = action(pol, b)
            r_total .+= d*rew_f(m,s,a)
            s, o = @gen(:sp,:o)(m, s, a)
            b = update(up,b,a,o)
            d *= discount(m)
        end
        push!(results,r_total)
    end
    mc_res = mean(results)
    mc_res_sem = 3 * std(results) / sqrt(runs)

    bel_val = pg_val
    #Compare and Report
    @info pg_res[1]==pg_res[2]
    @show mc_res
    @show bel_val
    # @show typeof(mc_res)
    # @show typeof(bel_val)
    is_pass = (abs.(mc_res-bel_val)<mc_res_sem)
    @info "Difference is $(mc_res-bel_val), 3 SEM is $mc_res_sem"
    @info "Passing: $is_pass"
    return is_pass
end

function pg_vs_mc(m::POMDP; solver=SARSOPSolver(;max_time=10.0),h=15,runs=5000)
    m_tuple = get_policy(m::POMDP; solver=solver)
    pg_res = belief_value_polgraph(m_tuple..., h)
    return compare_pg_rollout(m_tuple..., pg_res;h=500,runs=runs) #30000
end

function recur_vs_mc(m::POMDP; solver=SARSOPSolver(;max_time=10.0),h=15,runs=5000)
    m_tuple = get_policy(m::POMDP; solver=solver)
    pg_res = belief_value_recursive(m_tuple..., h)
    return compare_pg_rollout(m_tuple..., pg_res;h=h,runs=runs)
end

function multirew(m,s,a)
    # if isterminal(m,s)
    #     flag = 0
    # else
        flag = 1
    # end
    return vcat(reward(m,s,a), reward(m,s,a), flag)
end

function vector_test_pg(m::POMDP; solver=SARSOPSolver(;max_time=10.0),h=15,runs=10000)
    # @info m
    m_tuple = get_policy(m::POMDP; solver=solver)
    pg_res = belief_value_polgraph(m_tuple..., h;rewardfunction=multirew)
    # @info pg_res
    # s_one = sum([1*discount(m)^(x-1) for x in 1:h])
    # @info s_one
    # @info pg_res[3]
    # @info isapprox(s_one,pg_res[3];atol=0.0001)
    return pg_res[1]==pg_res[2] && compare_pg_rollout_vec(m_tuple..., pg_res,
            multirew,3;h=1000,runs=runs)
end

function vector_test_r(m::POMDP; solver=SARSOPSolver(;max_time=10.0),h=15,runs=10000)
    # @info m
    m_tuple = get_policy(m::POMDP; solver=solver)
    pg_res = belief_value_recursive(m_tuple..., h;rewardfunction=multirew)
    # @info pg_res
    # @info pg_res[1]==pg_res[2]
    s_one = sum([1*discount(m)^(x-1) for x in 1:h])
    @info s_one
    @info pg_res[3]
    @info isapprox(s_one,pg_res[3];atol=0.0001)
    return pg_res[1]==pg_res[2]&&compare_r_rollout_vec(m_tuple..., pg_res,
    multirew,3;h=h,runs=runs) 
    #&& isapprox(s_one,pg_res[3];atol=0.0001)
end

@testset "Policy Graph" begin
    testh = 55
    n_runs = 40000
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
    h=60
    runs=30000#50000
    m_tuple = get_policy(rs; solver=solver)
    pg_res = belief_value_polgraph(m_tuple..., h)
    @info pg_res[1]
    recur_res = belief_value_recursive(m_tuple..., h)[1]
    @info recur_res
    @show pg_res[1]-recur_res
    @test isapprox(pg_res[1],recur_res;atol=0.0001)
    @test compare_pg_rollout(m_tuple..., pg_res;h=500,runs=runs)
end

@testset "Vectorized Reward PG" begin
    testh=65
    nruns=2000
    @test vector_test_pg(rs;h=testh,runs=nruns)
    @test vector_test_pg(tiger;h=testh,runs=nruns)
    @test vector_test_pg(cb;h=testh,runs=nruns)
    @test vector_test_pg(mh;h=testh,runs=nruns)
    @test vector_test_pg(tm;h=testh,runs=nruns)
end

@testset "Vectorized Reward Recur" begin
    testh=10
    nruns=1000
    @test vector_test_r(rs;h=testh,runs=nruns)
    @test vector_test_r(tiger;h=testh,runs=nruns)
    @test vector_test_r(cb;h=testh,runs=nruns)
    @test vector_test_r(mh;h=testh,runs=nruns) #Why does this fail?
    @test vector_test_r(tm;h=testh,runs=nruns)
end

# @testset "GridWorldPOMDP" begin
#     h = 5
#     n_runs = 10000
#     @test pg_vs_mc(gw;h=testh,runs=n_runs)
#     @test recur_vs_mc(gw;h=testh,runs=n_runs)
# end

# crs = RockSampleCPOMDP()

# function testr(m,s,a)
#     vcat(reward(m,s,a), costs(m,s,a))
# end