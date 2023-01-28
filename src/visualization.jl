## Ben Kraske, bekr4901@colorado.edu, 11/15/2022

function display_graph(pg)
    n_nodes = length(pg.nodes)
    g = zeros(Int,n_nodes,n_nodes)
    edge_dict = Dict()
    name = ["first, $(pg.nodes[1])" string.(pg.nodes[2:end])...]
    for edge in pg.edges
        g[edge[1][1],edge[2]] = 1
        push!(edge_dict,(edge[1][1],edge[2])=>edge[1][2])
    end
    return graphplot(g,names=name, edgelabel=edge_dict)
end