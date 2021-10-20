using Base.Sys
using DFControl.Utils: searchdir
cd(assetfile("Fe/"))

big_files = searchdir(".", "bz2")
if islinux()
    for f in big_files
        run(`bunzip2 $f`)
    end
end

job = Job(".")
abgrid = DFW.AbInitioKGrid(job)

big_files = map(x -> splitext(x)[1], big_files)
if islinux()
    for f in big_files
        run(`bzip2 $f`)
    end
end
