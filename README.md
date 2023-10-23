# POMDPPolicyGraphs.jl
Policy Graphs for the POMDPs.jl Interface

## Methods

## Usage:

```julia

using POMDPs, RockSample, NativeSARSOP
using POMDPTools

pomdp = RockSamplePOMDP(5,7)
policy = solve(SARSOPSolver(;max_time=10.0),pomdp)
updater = DiscreteUpdater(pomdp)
b0 = initialize_belief(updater,initialstate(pomdp))

pg_evaluator = PolicyGraphEvaluator(pomdp,45)
pg_v = evaluate(pg_evaluator,pomdp,policy)
pg_v(b0)

ee_evaluator = ExhaustiveEvaluator(pomdp,45)
ee_v = evaluate(ee_evaluator,pomdp,policy)
ee_v(b0)

```