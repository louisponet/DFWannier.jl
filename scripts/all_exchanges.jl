using DFWannier

qe = load_job("/home/ponet/GdMn2O5/NSOC/qe/")
hamis = qe.local_dir .* DFControl.search_dir(qe.local_dir, "hr.dat")
wan_file = qe.local_dir * DFControl.search_dir(qe.local_dir, "wan_up.win")[1]
fermi = read_fermi_from_qe_file(qe.local_dir * search_dir(qe.local_dir,"scf.out")[1])
exchanges = WannExchanges(hamis[2], hamis[1], wan_file, fermi)
for i = 1:length(exchanges.infos)
    for j = i+1:length(exchanges.infos)
        print("exchange between $i,$j = $(exchange_between(i, j, exchanges))\n")
    end 
end
