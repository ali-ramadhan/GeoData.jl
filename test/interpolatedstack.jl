using GeoData, NCDatasets, ArchGDAL, Interpolations, Dates, Test

include(joinpath(dirname(pathof(GeoData)), "../test/test_utils.jl"))

@test calcfrac(DateTime(2001, 1, 1), DateTime(2001, 1, 2), DateTime(2001, 1, 1)) == 0
@test calcfrac(DateTime(2001, 1, 1), DateTime(2001, 1, 2), DateTime(2001, 1, 2)) == 1
@test calcfrac(DateTime(2001, 1, 1), DateTime(2001, 1, 2), DateTime(2001, 1, 1, 12)) == 0.5
@test calcfrac(1, 2, 1) == 0.0
@test calcfrac(1, 2, 1.5) == 0.5
@test calcfrac(1, 2, 2.0) == 1.0

@test Interpolations.value_weights(Cosine(), 0)[1] == 1.0
@test Interpolations.value_weights(Cosine(), 0.5)[1] ≈ 0.5
@test Interpolations.value_weights(Cosine(), 1)[1] == 0.0
@test Interpolations.value_weights(Cosine(), 0)[2] == 0.0
@test Interpolations.value_weights(Cosine(), 0.5)[2] ≈ 0.5
@test Interpolations.value_weights(Cosine(), 1)[2] == 1.0

@test Interpolations.value_weights(HypTan(), 0)[1] == 1.0
@test Interpolations.value_weights(HypTan(), 0.5)[1] ≈ 0.5
@test Interpolations.value_weights(HypTan(), 1)[1] == 0.0
@test Interpolations.value_weights(HypTan(), 0)[2] == 0.0
@test Interpolations.value_weights(HypTan(), 0.5)[2] ≈ 0.5
@test Interpolations.value_weights(HypTan(), 1)[2] == 1.0


int = interpolate([1, 2, 3, 4], BSpline(Cosine()))
@test int[2.0] == 2.0
@test int[2.25] < 2.75
@test int[2.25] == -cos(π * 2.25) / 2 + 2 + 0.5 
@test int[2.5] == 2.5
@test int[2.75] > 2.75
@test int[2.75] == -cos(π * 2.75) / 2 + 2 + 0.5 
@test int[3.0] == 3.0

# ncexamples = "https://www.unidata.ucar.edu/software/netcdf/examples/"
# ncmulti = geturl(joinpath(ncexamples, "test_echam_spectral.nc"))

# series = GeoSeries([NCDstack(ncmulti), NCDstack(ncmulti)], (Ti(DateTime(2020, 1):Month(1):DateTime(2020, 2)),); childtype=nothing)

# iseries = interpseries(series, DateTime(2020, 1):Day(1):DateTime(2020, 2), (ameltfrac=BSpline(Cosine()),))
# iseries2 = interpseries(series, DateTime(2020, 1):Day(1):DateTime(2020, 2), (ameltfrac=BSpline(Linear()),))
# iseries[1][:ameltfrac].frac
# iseries2[1][:ameltfrac]
# iseries[2].stacks[1].loaded
# iseries[10].stacks[2].loaded
# iseries[10][:srafs, 1:10, 1:10, 1]
# clear!(iseries[10])

gdalpath = geturl("https://download.osgeo.org/geotiff/samples/gdal_eg/cea.tif")
gdalstack = GDALstack((a=gdalpath, b=gdalpath))
geostack = GeoStack((a=0x02 .* GeoArray(GDALarray(gdalpath)), b=0x02 .* GeoArray(GDALarray(gdalpath))))
cstack1 = CachedStack(gdalstack)
cstack2 = CachedStack(geostack)
istack = InterpStack((cstack1, cstack2), (a=BSpline(Linear()), b=BSpline(Cosine()),), 1.25)
A = istack[:a]
B = istack[:b]
A = istack[:a, 1:10, 1:10, 1]
B = istack[:b, 1:10, 1:10, 1]
b = GeoArray(istack[:a]) 
B = istack[:b] |> plot
