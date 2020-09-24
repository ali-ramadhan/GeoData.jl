

"""
    CachedStack(parent; loaded=Dict{Symbol,Any}()) 

A stack that wraps another Stack and stores any layers loaded 
from it in a `Dict`. This prevents loading them multiple times
from disk.

It also means the are locked from the garbage collector if they are 
still in the Dict, which can cause problems with memory use.  Use 
`clear!(stack)` to return to the unloaded state, freeing memory.
"""
mutable struct CachedStack{S} <: AbstractGeoStack{S}
    stack::S
    cache::Dict{Symbol,Any}
end
CachedStack(stack; loaded=Dict{Symbol,Any}()) = 
    CachedStack(stack, loaded)

stack(s::CachedStack) = s.stack
cache(s::CachedStack) = s.cache

# GeoData methods

GeoData.refdims(s::CachedStack) = refims(stack(s))
GeoData.metadata(s::CachedStack) = metadata(stack(s))
GeoData.window(s::CachedStack) = window(stack(s))
GeoData.childtype(s::CachedStack) = childtype(stack(s))

# Base methods

Base.parent(s::CachedStack) = stack(s)
Base.keys(s::CachedStack) = keys(parent(s))
Base.values(s::CachedStack) = values(parent(s))
Base.names(s::CachedStack) = names(parent(s))
Base.length(s::AbstractGeoStack) = length(parent(s))

Base.getindex(stack::CachedStack, key::Symbol) = begin
    l = cache(stack)
    if haskey(l, key)
        l[key]
    else
        s = parent(stack)[key] |> GeoArray
        l[key] = s
        s
    end
end

# Clear

clear!(stack::MemGeoStack) = clear!(data(stack))
clear!(stack::DiskGeoStack) = nothing
clear!(stack::NamedTuple) = nothing
clear!(stack::CachedStack) = 
    for k in keys(cache(stack))
        delete!(cache(cache), k)
    end
