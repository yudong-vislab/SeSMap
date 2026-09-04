# Mapplet: a CORBA-based genome map viewer

## Motivation

There are a large number of genetic and physical maps, distributed at many sites. Each site offers different kinds of access methods and viewers. CORBA, the de facto standard for distributed object-oriented computing, offers new opportunities to unify the view on these maps through standard interfaces. A collaboration of Infobiogen and the EBI proposes a common IDL for maps.

## Results

A CORBA map viewer is presented which serves as a proof of concept for the proposed IDL. It demonstrates its usefulness in the context of map viewing and its ability to handle large maps with <1000 markers. The viewer gives access to the maps of the Radiation Hybrid Database at EBI. It gives a quick overview of several large maps side by side. The marker density at each map position is displayed and different marker types can be highlighted.

## Availability

Demonstration and source code at: http://sunny.ebi.ac.uk/ approximately jungfer/Mapplet
