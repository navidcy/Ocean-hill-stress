#Use Oceananigans v0.82. Small changes are required for newer versions.
#For other packages, any version from 2023 should suffice.

using Oceananigans,
      Oceananigans.Units
using Oceananigans.ImmersedBoundaries: immersed_peripheral_node
using CUDA
using QuadGK
using SpecialFunctions
using CairoMakie
Makie.inline!(true);

Nx, Ny, Nz = 350, 350, 100		#Number of grid cells in x, y and z dimensions

architecture = GPU()		#Can be changed to CPU if needed

name = "tdmixedh200"		#This code is for a mixed flow test with a 200m hill

const H  = 1.5kilometers		#Ocean depth
const Lx = 500kilometers		#Domain length along the x-axis
const Ly = 500kilometers		#Domain length along the y-axis

#The following code is used to create a non-linear grid spacing with
#faces xnew_faces, ynew_faces and znew_faces. This grid spacing creates a higher
#resolution near the hill. 

xscaler(x)=erf((x-(Lx/4))/(Lx/16))+erf((-x-(Lx/4))/(Lx/16))+3

yscaler(x)=erf((x-(Ly/4))/(Ly/16))+erf((-x-(Ly/4))/(Ly/16))+3

zscaler(x)=erf((x+(H/2))/(H/8))+2

xscale_sum=0
for k in 1:Nx
    global xscale_sum=xscale_sum+xscaler(-(Lx/2)+(k-1)*(Lx/Nx))
end

yscale_sum=0
for k in 1:Ny
    global yscale_sum=yscale_sum+yscaler(-(Ly/2)+(k-1)*(Ly/Ny))
end

zscale_sum=0
for k in 1:Nz
    global zscale_sum=zscale_sum+zscaler(-H+(k-1)*(H/Nz))
end

xscaling_factor=Lx/xscale_sum

yscaling_factor=Ly/yscale_sum

zscaling_factor=H/zscale_sum

xface_array=Array{Float64}(undef,Nx+1)
xface_array[1]=-(Lx/2)
for i in 2:(Nx+1)
    xface_array[i]=xface_array[i-1]+xscaling_factor*xscaler(-(Lx/2)+(i-2)*(Lx/Nx))
end

yface_array=Array{Float64}(undef,Ny+1)
yface_array[1]=-(Ly/2)
for i in 2:(Ny+1)
    yface_array[i]=yface_array[i-1]+yscaling_factor*yscaler(-(Ly/2)+(i-2)*(Ly/Ny))
end

zface_array=Array{Float64}(undef,Nz+1)
zface_array[1]=-H
for i in 2:(Nz+1)
    zface_array[i]=zface_array[i-1]+zscaling_factor*zscaler(-H+(i-2)*(H/Nz))
end

xnew_faces(k)=xface_array[trunc(Int,k)]
ynew_faces(k)=yface_array[trunc(Int,k)]
znew_faces(k)=zface_array[trunc(Int,k)]

#Definition of the grid without the hill

underlying_grid = RectilinearGrid(architecture,
                                  size = (Nx, Ny, Nz),
                                  x = xnew_faces,
                                  y = ynew_faces,
                                  z = znew_faces,
                                  halo = (4, 4, 4),
                                  topology = (Periodic, Periodic, Bounded))
								  

const h0 = 200 # m		#Height of hill in metres
const width = 5000 #m		#Half width of hill in metres
bump(x, y) = - H + h0 * exp(-(x^2+y^2) / 2width^2)		#Definition of the Gaussian hill

const Ni² = (4e-3)*(4e-3)  # [s⁻²] initial buoyancy frequency / stratification

println("Mount height: ",h0," Mount width: ",width)
println("Grid size: ",Nx,"x",Ny,"x",Nz)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bump))		#Definition of the grid with the hill

coriolis = FPlane(latitude =-20)		#The coriolis parameter 
const U_mean = 0.1		#The steady component of the flow in m/s
const mean_forcing_amplitude = (coriolis.f)*U_mean		#The forcing required to maintain the steady component of the flow

const T2 = 12.421hours		#The tidal period. This one is for the M2 tide	
const ω2 = 2π / T2 # radians/sec		#The tidal frequency
const U_tidal = 0.1		#The tidal component of the flow in m/s
const tidal_forcing_amplitude = U_tidal * (coriolis.f^2 - ω2^2) / ω2
@inline tidal_forcing(x, y, z, t) = tidal_forcing_amplitude * sin(ω2 * t)		#The forcing required to maintain the periodic component of the flow

Δt = 7.5 # seconds		#The time-step in seconds. Can be decreased if NANs are being produced.

max_Δt = Δt
free_surface = SplitExplicitFreeSurface(; grid, cfl = 0.7, max_Δt)

#In the following code we artificially increase the viscosity and stabilise buoyancy and velocity at the horizontal boundaries. 
#This also allows for one to maintain a steady flow in the case f=0 whereby the mean forcing amplitude is zero.

const ν1 = 1000
const ν2 = 1e-4
@inline variable_viscosity(x, y, z, t) = ν1*(erf((x-(Lx/4))/(Lx/16))+erf((-x-(Lx/4))/(Lx/16))+4+erf((y-(Ly/4))/(Ly/16))+erf((-y-(Ly/4))/(Ly/16)))
@inline buoyancy_stabiliser(x,y,z,t,b)=(z*Ni²-b)*ν2*(erf((x-(Lx/4))/(Lx/16))+erf((-x-(Lx/4))/(Lx/16))+4+erf((y-(Ly/4))/(Ly/16))+erf((-y-(Ly/4))/(Ly/16)))
@inline velocity_stabiliser(x,y,z,t,u)=(U_mean+U_tidal*cos(ω2 * t)-u)*ν2*(erf((x-(Lx/4))/(Lx/16))+erf((-x-(Lx/4))/(Lx/16))+4+erf((y-(Ly/4))/(Ly/16))+erf((-y-(Ly/4))/(Ly/16)))+tidal_forcing(x, y, z, t)
@inline mean_forcing(x, y, z, t) = mean_forcing_amplitude

horizontal_closure = HorizontalScalarDiffusivity(ν=variable_viscosity, κ=variable_viscosity)
buoyancy_forcing = Forcing(buoyancy_stabiliser, field_dependencies=:b)
velocity_forcing = Forcing(velocity_stabiliser, field_dependencies=:u)

#Definition of the closures and model

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    free_surface = free_surface,
                                    coriolis = coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO(),
                                    closure = (horizontal_closure,),
                                    forcing = (b=buoyancy_forcing,v=mean_forcing,u=velocity_forcing))
									
stop_time = 200hours		#How long the simulation is run for

simulation = Simulation(model, Δt=Δt, stop_time=stop_time)

using Printf

wall_clock = Ref(time_ns())

function print_progress(sim)

    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("iteration: %d, time: %s, wall time: %s, max|w|: %6.3e, m s⁻¹, next Δt: %s\n",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.velocities.w), prettytime(sim.Δt))

    wall_clock[] = time_ns()

    @info msg

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

#Defining variables which we track

b = model.tracers.b
u, v, w = model.velocities

U = Field(Average(u))

u′ = u - U

N² = ∂z(b)

S² = @at (Center, Center, Face) ∂z(u)^2 + ∂z(v)^2

Ri = N² / S²

#Computing the pressure in the bottom grid cell for eventual computation of drag

pressure =  model.pressure.pHY′
@inline bottom_condition(i, j, k, grid) = immersed_peripheral_node(i, j, k-1, grid, Center(), Center(), Center())
bottom_pressure = Field(Average(pressure, condition = bottom_condition, mask = 0.0, dims = 3))

output_interval=15minutes		#The interval at which data is outputted 		  

simulation.output_writers[:fields] = JLD2OutputWriter(model, (;U, bottom_pressure),
                                                      schedule = TimeInterval(output_interval),
                                                      with_halos = false,
                                                      filename = name,
                                                      overwrite_existing = true)
													  
# Initial conditions
ui(x, y, z) = U_mean+U_tidal

bi(x, y, z) = Ni² * z

println("Stratification: ",sqrt(Ni²))

set!(model, u=ui, v=0, b=bi)

#Running the simulation

run!(simulation)

#Extracting data from the simulation

saved_output_filename = name * ".jld2"

bottom_pressure_t = FieldTimeSeries(saved_output_filename, "bottom_pressure")
U_t=FieldTimeSeries(saved_output_filename, "U")

times = U_t.times

bumpderiv(x)=-( (h0*x) /width^2)*exp(-x^2 / 2width^2)		#Computing the derivative of the bump function
framesize=size(times)[1]

#Computing the stress at the bottom grid cell at each output interval

bottom_stress=Array{Float64}(undef,framesize,Nx,Ny)
for i in 1:framesize
    for j in 1:Nx
        for k in 1:Ny
            xint=quadgk(bumpderiv,xnew_faces(j),xnew_faces(j+1),rtol=1e-10)[1]
            bottom_stress[i,j,k]=(interior(bottom_pressure_t[i])[j,k])*quadgk(x-> xint*exp(-x^2 / 2width^2),xnew_faces(k),xnew_faces(k+1),rtol=1e-10)[1]
        end
    end
end

#Computing the total stress as a function of time

stress_t=Array{Float64}(undef,framesize)
for i in 1:framesize
    stress_t[i]=0
    for j in 1:Nx
        for k in 1:Ny    
            stress_t[i]=stress_t[i]+bottom_stress[i,j,k]
        end
    end
end

#Computing the average stress over the last 200 frames of stress

println("Mean stress final 200")
avg_stress=0
for i in (framesize-199):framesize
	global avg_stress=avg_stress+stress_t[i]
end
avg_stress=avg_stress/200
println(avg_stress)

#Computing the average stress over the last 100 frames of stress

println("Mean stress final 100")
avg_stress=0
for i in (framesize-99):framesize
	global avg_stress=avg_stress+stress_t[i]
end
avg_stress=avg_stress/100
println(avg_stress)

#Computing the average stress over the last 100-200 frames of stress

println("Mean stress final 100-200")
avg_stress=0
for i in (framesize-199):(framesize-99)
	global avg_stress=avg_stress+stress_t[i]
end
avg_stress=avg_stress/100
println(avg_stress)

#Computing the minimum stress over the last 100 frames of stress

println("Minimum stress")
minstress=stress_t[framesize-99]
minindex=framesize-99
for i in (framesize-99):framesize
    if stress_t[i]<minstress
        global minstress=stress_t[i]
		global minindex=i
    end
end
println(minstress)

#Computing the maximum stress over the last 100 frames of stress

println("Maximum stress")
maxstress=0
maxindex=0
for i in (framesize-99):framesize
    if stress_t[i]> maxstress
        global maxstress=stress_t[i]
		global maxindex=i
    end
end
println(maxstress)

#Computing the maximum velocity over the last 100 frames of stress

println("Maximum velocity")
maxvel=0
maxindex=0
for i in (framesize-99):framesize
    if U_t[i][1]> maxvel
        global maxvel=U_t[i][1]
		global maxindex=i
    end
end
println(maxvel)

#Printing values of the stress and velocity in the final frames for further processing

println("Final 200 frames of stress")
for i in (framesize-199):framesize
	print(stress_t[i],",")
end
println("")

println("Final 100 frames of stress")
for i in (framesize-99):framesize
	print(stress_t[i],",")
end
println("")

println("Final 100-200 frames of stress")
for i in (framesize-199):(framesize-100)
	print(stress_t[i],",")
end
println("")

println("Final 200 frames of velocity")
for i in (framesize-199):framesize
	print(U_t[i][1],",")
end
println("")

println("Final 100 frames of velocity")
for i in (framesize-99):framesize
	print(U_t[i][1],",")
end
println("")

println("Final 100-200 frames of velocity")
for i in (framesize-199):(framesize-100)
	print(U_t[i][1],",")
end
println("")

#Printing the values of the stress at all output intervals

println("All stress")
for i in 1:framesize
	println(stress_t[i])
end