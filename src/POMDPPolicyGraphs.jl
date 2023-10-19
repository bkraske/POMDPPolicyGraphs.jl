"""
Provides methods for generating and evaluating POMDP/MDP Policy Graphs
"""

module POMDPPolicyGraphs

using POMDPs
using POMDPTools
using LinearAlgebra
using SparseArrays
import NativeSARSOP

"""
    evaluate(pomdp::POMDP{S,A}, updater::Updater, pol::Policy, b::DiscreteBelief, depth::Int;rewardfunction=VecReward())

    Calculates the value of a policy recursively to a specified depth, calculating reward according to `rew_f``, the reward function passed.

"""
function evaluate end

include("recursive_evaluation.jl")
include("generation.jl")
include("evaluation.jl")

export
VecReward,
EvalTabularPOMDP,
PolicyGraphEvaluator,
ExhaustiveEvaluator,
evaluate,
gen_polgraph,
eval_polgraph,
gen_eval_polgraph,
calc_belvalue_polgraph,
belief_value_polgraph

end # module
