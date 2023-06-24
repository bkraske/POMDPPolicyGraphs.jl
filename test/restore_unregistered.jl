using Pkg
try Pkg.rm("ConstrainedPOMDPModels") catch end
Pkg.add(url="https://github.com/WhiffleFish/ConstrainedPOMDPModels")