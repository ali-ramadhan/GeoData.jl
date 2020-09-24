
using Test, Dates

spec = MinMaxSpec((tmin=Hour(5), tmax=Hour(14)), BSpline(Cosine()))

tfrac = calcfrac(DateTime(2001, 1, 1), DateTime(2001, 1, 2), DateTime(2001, 1, 1, 5))
@test windowfromtimes(spec, Day(1), tfrac) == ((tmin=1, tmax=1), 0.0)
tfrac = calcfrac(DateTime(2001, 1, 1), DateTime(2001, 1, 2), DateTime(2001, 1, 1, 14))
@test windowfromtimes(spec, Day(1), tfrac) == ((tmax=1, tmin=2), 0.0)

tfrac = calcfrac(DateTime(2001, 1, 1), DateTime(2001, 1, 2), DateTime(2001, 1, 1, 13, 59, 59, 99))
@test windowfromtimes(spec, Day(1), tfrac)[1] == (tmin=1, tmax=1)
@test windowfromtimes(spec, Day(1), tfrac)[2] ≈ 1.0 atol=1e4
tfrac = calcfrac(DateTime(2001, 1, 1), DateTime(2001, 1, 2), DateTime(2001, 1, 1, 4, 59, 59, 99))
@test windowfromtimes(spec, Day(1), tfrac)[1] == (tmax=0, tmin=1)
@test windowfromtimes(spec, Day(1), tfrac)[2] ≈ 1.0 atol=1e4
tfrac = calcfrac(DateTime(2001, 1, 1), DateTime(2001, 1, 2), DateTime(2001, 1, 2, 4, 59, 59, 99))
@test windowfromtimes(spec, Day(1), tfrac)[1] == (tmax=1, tmin=2)
@test windowfromtimes(spec, Day(1), tfrac)[2] ≈ 1.0 atol=1e4

tfrac = calcfrac(DateTime(2001, 1, 1), DateTime(2001, 1, 2), DateTime(2001, 1, 1, 8))
@test windowfromtimes(spec, Day(1), tfrac)[1] == (tmin=1, tmax=1) 
@test windowfromtimes(spec, Day(1), tfrac)[2] ≈ 1/3 
tfrac = calcfrac(DateTime(2001, 1, 1), DateTime(2001, 1, 2), DateTime(2001, 1, 1, 11))
@test windowfromtimes(spec, Day(1), tfrac)[1] == (tmin=1, tmax=1) 
@test windowfromtimes(spec, Day(1), tfrac)[2] ≈ 2/3 

tfrac = calcfrac(DateTime(2001, 1, 1), DateTime(2001, 1, 2), DateTime(2001, 1, 1, 19))
@test windowfromtimes(spec, Day(1), tfrac)[1] == (tmax=1, tmin=2) 
@test windowfromtimes(spec, Day(1), tfrac)[2] ≈ 1/3 
tfrac = calcfrac(DateTime(2001, 1, 1), DateTime(2001, 1, 2), DateTime(2001, 1, 1, 24))
@test windowfromtimes(spec, Day(1), tfrac)[1] == (tmax=1, tmin=2) 
@test windowfromtimes(spec, Day(1), tfrac)[2] ≈ 2/3 



# MinMax

gdalstack = GDALstack((a=gdalpath, b=gdalpath))
geostack = GeoStack((a=0x02 .* GeoArray(GDALarray(gdalpath)), b=0x02 .* GeoArray(GDALarray(gdalpath))))
cstack1 = CachedStack(gdalstack)
cstack2 = CachedStack(geostack)

trfrac = 1.25

mmfrac = MinMaxFracs(spec, Day(1), trfrac)
windowfromtimes(spec, Day(1), trfrac)
mmstack = MinMaxInterpStack((cstack1, cstack2), (a=BSpline(Linear()), b=BSpline(Cosine()),), trfrac, (temp=spec,))
mmtack[:temp]

# iseries = interpseries(series, DateTime(2020, 1):Day(1):DateTime(2020, 2), (ameltfrac=BSpline(Cosine()),))
# iseries2 = interpseries(series, DateTime(2020, 1):Day(1):DateTime(2020, 2), (ameltfrac=BSpline(Linear()),))
A = istack[:a]
B = istack[:b]
A = istack[:a, 1:10, 1:10, 1]
B = istack[:b, 1:10, 1:10, 1]
b = GeoArray(istack[:a]) 
B = istack[:b]

