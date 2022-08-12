# Based on documentation of DFTK
using LibGit2: LibGit2
using Pkg: Pkg

# To manually generate the docs:
#
# 1. Install all python dependencies from the PYDEPS array below.
# 2. Run "julia make.jl" to generate the docs

# Set to true to disable some checks and cleanup
DEBUG = true

# Where to get files from and where to build them
SRCPATH = joinpath(@__DIR__, "src")
BUILDPATH = joinpath(@__DIR__, "build")
ROOTPATH = joinpath(@__DIR__, "..")
CONTINUOUS_INTEGRATION = get(ENV, "CI", nothing) == "true"

# Python and Julia dependencies needed for running the notebooks
# PYDEPS = ["ase", "pymatgen"]
JLDEPS = [Pkg.PackageSpec(; url = "https://github.com/louisponet/DFWannier.jl.git",
                          rev = LibGit2.head(ROOTPATH))]

# Setup julia dependencies for docs generation if not yet done
Pkg.activate(@__DIR__)
# if !isfile(joinpath(@__DIR__, "Manifest.toml"))
    # Pkg.develop(Pkg.PackageSpec(; path = ROOTPATH))
    # Pkg.instantiate()
    # Pkg.add(; url = "https://github.com/kimikage/Documenter.jl", rev = "ansicolor")
# end

# Setup environment for making plots
ENV["GKS_ENCODING"] = "utf8"
ENV["GKSwstype"] = "100"
ENV["PLOTS_TEST"] = "true"

# Import packages for docs generation
using UUIDs
using DFWannier
using Documenter
using Literate

# Collect examples from the example index (src/index.md)
# The chosen examples are taken from the examples/ folder to be processed by Literate
EXAMPLES = [String(m[1])
            for m in
                match.(r"\"(examples/[^\"]+.md)\"",
                       readlines(joinpath(SRCPATH, "index.md"))) if !isnothing(m)]

# Collect files to treat with Literate (i.e. the examples and the .jl files in the docs)
# The examples go to docs/literate_build/examples, the .jl files stay where they are
literate_files = NamedTuple{(:src, :dest, :example),Tuple{String,String,Bool}}[(src = joinpath(ROOTPATH,
                                                                                               splitext(file)[1] *
                                                                                               ".jl"),
                                                                                dest = joinpath(SRCPATH,
                                                                                                "examples"),
                                                                                example = true)
                                                                               for file in
                                                                                   EXAMPLES]
for (dir, directories, files) in walkdir(SRCPATH)
    for file in files
        if endswith(file, ".jl")
            push!(literate_files, (src = joinpath(dir, file), dest = dir, example = false))
        end
    end
end
if !isempty(literate_files)
    
    s = Server(name=gethostname(), port=8080, domain = "localhost", scheduler = Servers.Bash(), uuid = string(uuid4()), julia_exec=Sys.BINDIR * "/julia")
    if !exists(s)
        Servers.initialize_config_dir(s)
        save(s)
    else
        s = Servers.local_server()
    end
    if !isalive(s)
        @info "Starting server here"
        Servers.initialize_config_dir(s)
        @async DFC.Resource.run()
    end
end
# Run Literate on them all
for file in literate_files
    # preprocess = file.example ? add_badges : identity
    # Literate.markdown(file.src, file.dest; documenter=true, credit=false,
    #                   preprocess=preprocess)
    # Literate.notebook(file.src, file.dest; credit=false,
    #                   execute=CONTINUOUS_INTEGRATION || DEBUG)
    Literate.markdown(file.src, file.dest; documenter = true, credit = false)
    Literate.notebook(file.src, file.dest; credit = false,
                      execute = CONTINUOUS_INTEGRATION || DEBUG)
end

# Generate the docs in BUILDPATH
makedocs(; modules = [DFWannier],
         format = Documenter.HTML(
                                  # Use clean URLs, unless built as a "local" build
                                  ; prettyurls = CONTINUOUS_INTEGRATION,
                                  canonical = "https://louisponet.github.io/DFWannier.jl/stable/",
                                  assets = ["assets/favicon.ico"]),
         sitename = "DFWannier.jl", authors = "Louis Ponet", linkcheck = false,  # TODO
         linkcheck_ignore = [
                             # Ignore links that point to GitHub's edit pages, as they redirect to the
                             # login screen and cause a warning:
                             r"https://github.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/edit(.*)"],
         pages = ["Home" => "index.md",
                  "Tight Binding" => "tight_binding.md", 
                  # "Getting started" => Any["guide/installation.md",
                  #                          "guide/configuration.md",
                  #                          "Basic Tutorial"=>"guide/basic_tutorial.md",
                  #                          "Advanced Tutorial"=>"guide/advanced_tutorial.md"],
                  "Exchanges" => "exchanges.md",
                  "Berry" => "berry.md",
                  # "Examples" => EXAMPLES,
                  # "api.md"
                  # "publications.md",
                  ]
         # strict = !DEBUG,
         )

# Dump files for managing dependencies in binder
if CONTINUOUS_INTEGRATION
    cd(BUILDPATH) do
        open("environment.yml", "w") do io
            return print(io, """
                             name: dfwannier
                             """)
        end

        # Install Julia dependencies into build
        Pkg.activate(".")
        for dep in JLDEPS
            Pkg.add(dep)
        end
    end
    Pkg.activate(@__DIR__)  # Back to Literate / Documenter environment
end

# Deploy docs to gh-pages branch
deploydocs(; repo = "github.com/louisponet/DFWannier.jl.git")

# Remove generated example files
if !DEBUG
    for file in literate_files
        base = splitext(basename(file.src))[1]
        for ext in [".ipynb", ".md"]
            rm(joinpath(file.dest, base * ext); force = true)
        end
    end
end

if !CONTINUOUS_INTEGRATION
    println("\nDocs generated, try $(joinpath(BUILDPATH, "index.html"))")
end
