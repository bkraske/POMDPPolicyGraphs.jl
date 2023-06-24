using Pkg
try Pkg.rm("ConstrainedPOMDPs") catch end
Pkg.add(url="https://github.com/qhho/ConstrainedPOMDPs.jl")
try Pkg.rm("ConstrainedPOMDPModels") catch end
Pkg.add(url="https://github.com/WhiffleFish/ConstrainedPOMDPModels")