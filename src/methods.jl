"""
    replace_missing(a::AbstractGeoArray, newmissingval)
    replace_missing(a::AbstractGeoStack, newmissingval)

Replace missing values in the array or stack with a new missing value, 
also updating the `missingval` field/s.
"""
replace_missing(a::DiskGeoArray, args...) = 
    replace_missing(GeoArray(a), args...)
replace_missing(a::MemGeoArray, newmissingval=missing) = begin
    newdata = if ismissing(missingval(a))
        collect(Missings.replace(parent(a), newmissingval))
    else
        replace(parent(a), missingval(a) => newmissingval)
    end
    rebuild(a; data=newdata, missingval=newmissingval)
end
replace_missing(stack::AbstractGeoStack, newmissingval=missing) = 
    rebuild(stack, map(a -> replace_missing(a, newmissingval, values(stack))))

"""
    boolmask(A::AbstractArray, [missingval])

Create a mask array of `Bool` values, from any AbstractArray. For `AbstractGeoArray` 
the default `missingval` is `missingval(A)`, for all other `AbstractArray`s 
it is `missing`.

The array returned from calling `boolmask` on a `AbstractGeoArray` is a 
[`GeoArray`](@ref) with the same size and fields as the oridingl array
"""
function boolmask end

boolmask(A::AbstractArray) = boolmask(A, missingval(A))
boolmask(A::AbstractGeoArray, missingval) = _boolmask(A, missingval)
boolmask(A::AbstractGeoArray, ::Missing) = _boolmask(A, missing)
# Need to catch NaN and missing with === as isapprox will miss them
boolmask(A::AbstractArray, missingval) =
    (x -> !(isapprox(x, missingval) || x === missingval)).(parent(A))
boolmask(A::AbstractArray, ::Missing) = (x -> x !== missing).(parent(A))

# Avoids ambiguity with AbstractArray methods
_boolmask(A::AbstractGeoArray, missingval) =
    rebuild(A; data=boolmask(parent(A), missingval), missingval=false, name=:boolmask)

"""
    missingmask(A::AbstractArray, [missingval])

Create a mask array of `missing` or `true` values, from any AbstractArray. 
For `AbstractGeoArray` the default `missingval` is `missingval(A)`, 
for all other `AbstractArray`s it is `missing`.

The array returned from calling `boolmask` on a `AbstractGeoArray` is a 
[`GeoArray`](@ref) with the same size and fields as the oridingl array
"""
function missingmask end
missingmask(A::AbstractArray) = missingmask(A, missingval(A))
missingmask(A::AbstractGeoArray, missingval) = _missingmask(A, missingval)
missingmask(A::AbstractGeoArray, missingval::Missing) = _missingmask(A, missingval)
missingmask(A::AbstractArray, missingval) =
    (x -> isapprox(x, missingval) || x === missingval ? missing : true).(parent(A))
missingmask(A::AbstractArray, missingval::Missing) =
    (x -> x === missingval ? missing : true).(parent(A))

# Avoids ambiguity with AbstractArray methods
_missingmask(A::AbstractGeoArray, missingval) =
    rebuild(A; data=missingmask(parent(A), missingval), missingval=missing, name=:missingmask)
