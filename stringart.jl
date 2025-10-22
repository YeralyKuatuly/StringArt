module StringArt

using Base.Threads: @threads, nthreads
using Dates
using FileIO
using Images
using LRUCache
using Logging
using Printf
using Random
using Statistics

const Point = ComplexF64
const Chord = Pair{Point,Point}
const GrayImage = Matrix{N0f8}
const RGBColor = RGB{N0f8}
const RGBImage = Matrix{RGBColor}
const Colors = Vector{RGBColor}
const DefaultArgs = Dict{String,Any}

const GIF_INTERVAL = 50
const RANDOMIZED_PIN_INTERVAL = 100
const SMALL_CHORD_CUTOFF = 0.10
const EXCLUDE_REPEATED_PINS = false

const lru = LRU{Chord, GrayImage}(maxsize=10000)

@enum StringArtMode GrayscaleMode RgbMode PaletteMode

export StringArtMode
export load_image
export run
export save_gif
export save_svg
export loghelper

# debug functions
export plot_pins
export plot_chords
export plot_color

mutable struct GifWrapper
    frames::Array{RGBColor}
    count::Int
end

# A wrapper that dispatches based on the OperationType
function load_image(image_path::String, size::Int, colors::Colors, mode:: StringArtMode)::Vector{GrayImage}
    load_image(image_path, size, colors, Val(mode))
end

""" Load and preprocess a grayscale image: crop to square and resize. """
function load_image(image_path::String, size::Int, colors::Colors, mode:: Val{GrayscaleMode})::Vector{GrayImage}
    # Read the image and convert it to an array
    @assert isfile(image_path) "Image file not found: $image_path"
    img = Images.load(image_path)
    # Resize the image to the specified dimensions
    img = crop_to_square(img)
    img = Images.imresize(img, size, size)
    # Convert the Image to gray scale and wrap in a vector
    return [convert(Matrix{N0f8}, Gray{N0f8}.(img))]
end

""" Load and decompose color image into grayscale channels based on given RGB filters. """
function load_image(image_path::String, size::Int, colors::Colors, mode:: Val{RgbMode})::Vector{GrayImage}
    # Read the image and convert it to an array
    @assert isfile(image_path) "Image file not found: $image_path"
    img = Images.load(image_path)
    # Resize the image to the specified dimensions
    img = crop_to_square(img)
    img = Images.imresize(img, size, size)
    return [red.(img), green.(img), blue.(img)]
end

""" Load an image and create a set of grayscale images representing how well each pixel matches the colors in the palette. """
function load_image(image_path::String, size::Int, colors::Colors, mode:: Val{PaletteMode})::Vector{GrayImage}
    # Read the image and convert it to an array
    @assert isfile(image_path) "Image file not found: $image_path"
    img = Images.load(image_path)
    # Resize the image to the specified dimensions
    img = crop_to_square(img)
    img = Images.imresize(img, size, size)

    # Convert the Image and colors to Lab space for better color comparison
    img = convert.(Lab{Float64}, img)
    colors = convert.(Lab{Float64}, colors)

    function distance(pixel_lab::Lab{Float64}, target_lab::Lab{Float64})::Float64
        return sqrt((pixel_lab.l - target_lab.l)^2 + (pixel_lab.a - target_lab.a)^2 + (pixel_lab.b - target_lab.b)^2)
    end

    # Create separate grayscale layer for each color
    layers = [zeros(Float64, size, size) for _ in eachindex(colors)]
    for (idx, color) in enumerate(colors)
        layers[idx] = clamp01nan.(10.0 ./ (1.0 .+ distance.(img, color)))
    end
    return [complement.(layer) for layer in layers]
end

""" Crop rectangular image to a centered square. """
function crop_to_square(image::Matrix)::Matrix
    # Calculate the size of the square
    height, width = size(image)
    crop_size = min(height, width)
    # Calculate the starting coordinates for the crop
    start_h = div(height - crop_size, 2) + 1
    start_w = div(width - crop_size, 2) + 1
    # Crop the image to a square shape
    @views image[start_h:(start_h+crop_size-1), start_w:(start_w+crop_size-1)]
end

""" Main function to generate string art image. Returns final image in png, svg and gif formats. """
function run(input::Vector{GrayImage}, args::DefaultArgs)::Tuple{RGBImage,String,GifWrapper}
    # generate all chords to be draw in the canvas
    chords = Tuple[]
    chords_with_pins = Tuple[]
    for (color, img) in zip(args["colors"], input)
        @info loghelper(args) * "Iterating image with color: #$(hex(color))"
        for (chord, from_pin, to_pin) in run_algorithm(img, args) 
            push!(chords, (chord, color))
            push!(chords_with_pins, (chord, color, from_pin, to_pin))
        end
    end
    if args["svg"]
        save_pin_sequence(chords_with_pins, args)
    end
    shuffle!(chords)

    # initialize output image
    png = zeros(RGBColor, args["size"], args["size"])
    # create struct that holds gif frames
    gif = gen_gif_wrapper(args)
    # initialize svg content
    svg = [svg_header(args)]
    # initialize an image for each color
    images = Dict(color=>zeros(N0f8, args["size"], args["size"]) for color in args["colors"])

    @info loghelper(args) * "Rendering Chords"
    for (n, (chord, color)) in enumerate(chords)
        # add chord to png image
        img = gen_img(chord, args)
        add_imgs!(images[color], img)

        # draw svg shape
        if args["svg"]
            push!(svg, draw_line(chord, color, args))
        end
        # save gif frame
        if args["gif"] && n % GIF_INTERVAL == 0
            png = join_channels(images, args["mode"])
            save_frame(png, gif)
        end
    end
    push!(svg, "</svg>")

    png = join_channels(images, args["mode"])
    return (png, join(svg, "\n"), gif)
end


# A wrapper that dispatches based on the OperationType
function join_channels(images::Dict{RGBColor,GrayImage}, mode:: StringArtMode)::RGBImage
    return join_channels(images, Val(mode))
end

""" Join GrayImage channels into a single RGBImage based on the specified mode. """
function join_channels(images::Dict{RGBColor,GrayImage}, mode::Val{GrayscaleMode})::RGBImage
    return complement.(images[RGB(0,0,0)] .* RGB(1,1,1))
end

""" Join RGB channels into a single RGBImage. """
function join_channels(images::Dict{RGBColor,GrayImage}, mode::Val{RgbMode})::RGBImage
    r, g, b = images[RGB(1,0,0)], images[RGB(0,1,0)], images[RGB(0,0,1)]
    return complement.(RGB.(r, g, b))
end

""" Add multiple color channels to create a composite RGB image with proper color blending. """
function join_channels(images::Dict{RGBColor,GrayImage}, mode::Val{PaletteMode})::RGBImage
    # Get dimensions from the first image
    height, width = size(first(values(images)))

    r = zeros(Float32, height, width)
    g = zeros(Float32, height, width)
    b = zeros(Float32, height, width)

    # Add each color layer with its respective intensity
    for (color, img) in images
        color = complement.(color)
        for i in eachindex(img)
            intensity = Float64(img[i])
            r[i] += intensity * color.r
            b[i] += intensity * color.b
            g[i] += intensity * color.g
        end
    end

    # Normalize the RGB values to the range [0, 1]
    max_val = max(quantile(vec(r), 0.99), quantile(vec(g), 0.99), quantile(vec(b), 0.99), 1)
    r ./= max_val
    g ./= max_val
    b ./= max_val
    return complement.(RGB.(clamp01nan.(r), clamp01nan.(g), clamp01nan.(b)))
end

""" Core string art generation loop. Produces ordered chords for image approximation. """
function run_algorithm(input::GrayImage, args::DefaultArgs)::Vector{Tuple{Chord, Point, Point}}
    steps = div(args["steps"], length(args["colors"]))

    @debug "Generating chords and pins positions"
    output = Vector{Tuple{Chord, Point, Point}}()

    pins = gen_pins(args["pins"], args["size"])
    pin2chords = Dict(p => gen_chords(p, pins, args["size"]) for p in pins)

    @debug "Starting algorithm..."
    pin = rand(pins)
    for step = 1:steps
        @debug "Step: $step"
        if step % RANDOMIZED_PIN_INTERVAL == 0
            pin = rand(pins)
        end

        @debug "Generating chord images..."
        chords = pin2chords[pin]

        if EXCLUDE_REPEATED_PINS && length(chords) == 0
            @debug "No chords left, breaking..."
            break
        end

        @debug "Calculating error in chords..."
        error, idx = select_best_chord(input, chords, args)
        chord, img = chords[idx], gen_img(chords[idx], args)
        @debug "Error calculated" idx, error

        @debug "Updating images and position..."
        add_imgs!(input, img)
        
        next_pin = (chord.first == pin) ? chord.second : chord.first
        push!(output, (chord, pin, next_pin))

        # don't draw the same chord again
        EXCLUDE_REPEATED_PINS && filter!(c -> c != chord, pin2chords[pin])
        # use the second point of the chord as the next pin
        pin = next_pin
    end
    return output
end

""" Generate `n` evenly spaced points around a circle on a square canvas. """
function gen_pins(pins::Int, size::Int)::Vector{Point}
    center = (size / 2) + (size / 2) * 1im
    radius = 0.95 * (size / 2)
    # divide the circle into n_points
    interval = 360 / pins
    # calc polar coordinates
    phi = deg2rad.(0:interval:360)
    coords = radius .* exp.(phi .* 1im)
    # add center to coords and round the values
    return round.(coords .+ center) |> unique
end

""" Generate valid chords from a given point `p` to other canvas points. """
function gen_chords(p::Point, points::Vector{Point}, size::Int)::Vector{Chord}
    # exclude small chords
    threshold = size * SMALL_CHORD_CUTOFF
    # line connecting a point to all other neighbors / canvas pins
    return [to_chord(p, q) for q in points if abs(p - q) > threshold]
end

""" Create an ordered chord (pair of points). """
function to_chord(p::Point, q::Point)::Chord
    # pair should be ordered so it can be searched
    p, q = sort([p, q], by=x -> (real(x), imag(x)))
    return Pair(p, q)
end

""" Find best chord that minimizes difference to target image. """
function select_best_chord(img::GrayImage, chords::Vector{Chord}, args::DefaultArgs)::Tuple{Float64,Int}
    nchords = length(chords)
    # Create more balanced chunks based on available threads
    chunks = [i:min(i + div(nchords, nthreads()) - 1, nchords) for i in 1:div(nchords, nthreads()):nchords]

    # Pre-allocate error array
    cimg = complement.(img)
    errors = fill(Inf32, length(chords))

    # Parallelize the error calculation
    @threads for t in eachindex(chunks)
        for i in chunks[t]
            chord_img = gen_img(chords[i], args)
            @inbounds errors[i] = @fastmath Images.ssd(cimg, chord_img)
        end
    end
    return findmin(errors)
end

""" Generate grayscale image representing a line between two points. """
function gen_img(chord::Chord, args::DefaultArgs)::GrayImage
    get!(lru, chord) do
        # extract parameters from cli args
        size, blur = args["size"], args["blur"]
        strength = convert(N0f8, args["line-strength"] / 100)

        # get endpoints
        p, q = chord

        # create an empty image with pre-allocated zeros
        m = zeros(Gray{N0f8}, size, size)

        # Draw the line using Bresenham's algorithm
        bresenham_line!(m,
                        round(Int, real(p)), round(Int, imag(p)),
                        round(Int, real(q)), round(Int, imag(q)),
                        strength)

        # gaussian filter to smooth the line
        return imfilter(m, Kernel.gaussian(blur))
    end
end

""" Draw a line between two points using Bresenham's line algorithm. """
function bresenham_line!(img::Matrix{Gray{N0f8}}, x0::Int, y0::Int, x1::Int, y1::Int, strength::N0f8)
    # Ensure coordinates are within image bounds
    height, width = size(img)
    x0 = clamp(x0, 1, width)
    y0 = clamp(y0, 1, height)
    x1 = clamp(x1, 1, width)
    y1 = clamp(y1, 1, height)

    # Calculate line parameters
    steep = abs(y1 - y0) > abs(x1 - x0)

    # If the line is steep, transpose the image and coordinates
    if steep
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    end

    # Ensure x0 <= x1
    if x0 > x1
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    end

    # Calculate deltas and initial error
    dx = x1 - x0
    dy = abs(y1 - y0)
    err = div(dx, 2)

    # Determine step direction
    y_step = y0 < y1 ? 1 : -1
    y = y0

    # Draw the line pixel by pixel
    for x in x0:x1
        # If steep, plot (y,x) instead of (x,y)
        if steep
            if 1 <= y <= width && 1 <= x <= height
                @inbounds img[x, y] = strength
            end
        else
            if 1 <= x <= width && 1 <= y <= height
                @inbounds img[y, x] = strength
            end
        end

        # Update error and possibly y coordinate
        err -= dy
        if err < 0
            y += y_step
            err += dx
        end
    end

    return img
end

""" Safely inplace add two grayscale images, handling overflow for N0f8 values. """
function add_imgs!(dst::GrayImage, src::GrayImage)
    @fastmath @inbounds @simd for i in eachindex(dst, src)
        val = Float32(dst[i]) + Float32(src[i])
        dst[i] = val > 1.0f0 ? Gray{N0f8}(1.0f0) : Gray{N0f8}(val)
    end
    return dst
end


""" Add a frame to the gif sequence. """
function save_frame(img::RGBImage, gif::GifWrapper)
    gif.frames[:, :, gif.count] .= img
    gif.count += 1
end

""" Write gif frames to disk. """
function save_gif(output::String, gif::GifWrapper)
    gif_frames = gif.frames[:, :, 1:(gif.count-1)]
    save(output * ".gif", gif_frames, fps=5)
end

""" Initialize gif wrapper for given step count and color mode. """
function gen_gif_wrapper(args::Dict)::GifWrapper
    if !args["gif"]
        return GifWrapper(Array{RGBColor}(undef, 0, 0, 0), 0)
    end
    n_colors = length(args["colors"])
    n_frames = n_colors * div(args["steps"], GIF_INTERVAL)
    frames = Array{RGBColor}(undef, args["size"], args["size"], n_frames)
    return GifWrapper(frames, 1)
end

""" Generate SVG header with specified size. """
function svg_header(args::DefaultArgs)::String
    size = args["size"]
    blur = args["blur"]
    return """<svg xmlns="http://www.w3.org/2000/svg" width="$size" height="$size" viewBox="0 0 $size $size">
    <filter id="blur">
        <feGaussianBlur stdDeviation="$blur" />
    </filter>"""
end

""" Draw a line in SVG format. """
function draw_line(chord::Chord, color::RGBColor, args::DefaultArgs)::String
    x1, y1 = real(chord.first), imag(chord.first)
    x2, y2 = real(chord.second), imag(chord.second)
    width = @sprintf("%.2f", args["line-strength"] / 100)
    return """<line x1="$x1" x2="$x2" y1="$y1" y2="$y2" stroke="#$(hex(color))" stroke-width="$width" filter="url(#blur)"/>"""
end

""" Write svg to disk. """
function save_svg(output::String, svg::String)
    open(output * ".svg", "w") do f
        write(f, svg)
    end
end

### DEBUGGING UTILITIES

""" Visual debug: overlay pin locations on image. """
function plot_pins(input::GrayImage, args::DefaultArgs)::GrayImage
    LEN = 4

    @debug "Generating pins positions"
    pins = gen_pins(args["pins"], args["size"])

    @debug "Plotting pins positions"
    width, height = size(input)
    for pin in pins
        lbx, ubx = Int(max(real(pin) - LEN, 0)), Int(min(real(pin) + LEN, width))
        lby, uby = Int(max(imag(pin) - LEN, 0)), Int(min(imag(pin) + LEN, height))
        input[lbx:ubx, lby:uby] .= 0
    end

    @debug "Done"
    return input
end

""" Visual debug: draw all chords from the first pin. """
function plot_chords(input::GrayImage, args::DefaultArgs)::GrayImage
    @debug "Generating chords"
    pins = gen_pins(args["pins"], args["size"])
    chords = gen_chords(pins[1], pins, args["size"])

    @debug "Plotting chords"
    for chord in chords
        add_imgs!(input, gen_img(chord, args))
    end

    @debug "Done"
    return clamp01nan.(input)
end

""" Visual debug: returns first grayscale channel. Stub for color support. """
function plot_color(input::Vector{GrayImage}, args::DefaultArgs)::GrayImage
    return input[1]
end

function loghelper(args::DefaultArgs)
    now = Dates.now()
    total = abs(Dates.value(now - get!(args, "start_elapsed", now)) / 1e3)
    delta = abs(Dates.value(now - get!(args, "last_log_elapsed", now)) / 1e3)
    args["last_log_elapsed"] = now

    return @sprintf("%6.2fs (%+5.2fs) => ", total, delta)
end

""" Save the continuous pin sequence to a text file """
function save_pin_sequence(chords_with_pins::Vector{Tuple}, args::DefaultArgs)
    output_path = args["output"]
    
    open(output_path * "_sequence.txt", "w") do f
        write(f, "STRING ART BUILD INSTRUCTIONS\n")
        write(f, "=" ^ 60 * "\n\n")
        write(f, "Total nails: $(args["pins"])\n")
        write(f, "Total connections: $(length(chords_with_pins))\n")
        write(f, "Estimated thread length: ~$(Int(round(length(chords_with_pins) * 0.5)))m\n\n")
        write(f, "CONTINUOUS PATH:\n")
        write(f, "=" ^ 60 * "\n\n")
        
        # Convert complex coordinates to nail numbers
        function coord_to_nail(point::Point)::Int
            center = (args["size"] / 2) + (args["size"] / 2) * 1im
            ang = Base.angle(point - center)
            angle_deg = (rad2deg(ang) + 360) % 360
            nail = Int(floor((angle_deg / 360) * args["pins"]))
            return nail % args["pins"]
        end
        
        # Now we have the actual from_pin and to_pin!
        for (i, (chord, color, from_pin, to_pin)) in enumerate(chords_with_pins)
            nail_from = coord_to_nail(from_pin)
            nail_to = coord_to_nail(to_pin)
            
            if i == 1
                write(f, "START: Tie thread to Nail $nail_from\n\n")
            end
            
            write(f, "Step $(lpad(i, 4)): From Nail $(lpad(nail_from, 3)) → to Nail $(lpad(nail_to, 3))\n")
        end
        
        # Get last nail
        last_chord, last_color, last_from, last_to = chords_with_pins[end]
        last_nail = coord_to_nail(last_to)
        
        write(f, "\nEND: Tie off at Nail $last_nail\n")
        write(f, "\n" * "=" ^ 60 * "\n")
        write(f, "HOW TO BUILD:\n")
        write(f, "=" ^ 60 * "\n")
        write(f, "1. Arrange $(args["pins"]) nails evenly around a circle\n")
        write(f, "2. Number them 0-$(args["pins"]-1) (clockwise from top)\n")
        write(f, "3. Tie thread to START nail\n")
        write(f, "4. Follow sequence - wrap once around each nail\n")
        write(f, "5. NEVER CUT THREAD - one continuous string!\n")
        write(f, "6. Tie off at END nail\n")
    end
    
    @info loghelper(args) * "Saved pin sequence to $(output_path)_sequence.txt"
end

end
