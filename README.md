# nearbyNuggets
Code and utilities used for analysis of MADCASH data -- especially searching for dwarf satellite "nuggets."

For best results, use this while listening to the [original Nuggets compilation album](https://en.wikipedia.org/wiki/Nuggets:_Original_Artyfacts_from_the_First_Psychedelic_Era,_1965%E2%80%931968).

Modules to implement:

inputCatalogs:
- X Read photometric catalogs
- X Add information to the catalogs (extinction, etc.)
- X Filter the catalogs (quality cuts, RGB selection, etc.)
    - O RGB "box" filter (written, but move from inputCatalogs to utils?)
    - X Isochrone filtering (in toolbox/utils)
    - O Matched filtering
- X Calculate median magnitude error in magnitude bins

mining:
- X Calculate binned number counts on the sky (and related statistics)
    - O Scale stars by their completeness weights to create a "normalized" density map (in progress)
- X Detect candidate overdensities
- O Select candidate globular clusters
    - O At distances where Gaia is useful
    - O At any distance (not requiring Gaia)

analysis:
- X Candidate diagnostic plots
  - suggestions for diagnostic plots: overlay point sources on color image, add a zoomed-in color image, eliminate g/i bands?, add luminosity function (with expected theoretical LF?)

- X Estimate structural parameters for candidates
  - X Max likelihood
  - X MCMC
- O Estimate luminosities for candidates (in progress - just needs clean-up)
- O TRGB distance estimates
- O (Isochrone-based) [Fe/H] estimates
- O Fit/overplot surface density profiles
    - O Sersic, exponential, King model fits

foolsGold:
- O Create catalog of fake stars to inject
- O Photometric completeness, bias, etc. from AST analysis
- O Create catalog of fake dwarfs to inject

other:
- X Read and overplot isochrones
    - O Currently only has a single set of 10 Gyr Padova isochrones. Add other ages (and isochrone sets)?
- O CMD plotting function? Other plotting functions?
- X Image cutouts
- O Surface/aperture photometry?
