# nearbyNuggets
Code and utilities used for analysis of MADCASH data -- especially searching for dwarf satellite "nuggets."

For best results, use this while listening to the [original Nuggets compilation album](https://en.wikipedia.org/wiki/Nuggets:_Original_Artyfacts_from_the_First_Psychedelic_Era,_1965%E2%80%931968).

Modules to implement:
- X Read photometric catalogs
- X Add information to the catalogs (extinction, etc.)
- X Filter the catalogs (quality cuts, RGB selection, etc.)
    - O Isochrone filtering
- X Calculate median magnitude error in magnitude bins
- O Read and overplot isochrones
- O CMD plotting function? Other plotting functions?
- O Matched filtering
- X Calculate binned number counts on the sky (and related statistics)
- O Candidate diagnostic plots
- O Estimate structural parameters for candidates
- O Estimate luminosities for candidates
- O TRGB distance estimates
- O (Isochrone-based) [Fe/H] estimates
- O Generate fake dwarfs
