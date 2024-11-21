struct PolicyGraphBelief
    node::Int
end

struct PolicyGraphUpdater <: POMDPs.Updater
    graph::PolicyGraph
end

function POMDPs.initialize_belief(updater::PolicyGraphUpdater,initialstate)
    return PolicyGraphBelief(updater.graph.node1)
end

function POMDPs.update(updater::PolicyGraphUpdater,b::PolicyGraphBelief,a,o)
    @assert a == updater.graph.nodes[b.node]
    return PolicyGraphBelief(updater.graph.edges[(b.node,o)])
end

POMDPs.action(g::PolicyGraph,b::PolicyGraphBelief) = g.nodes[b.node]