#############################################################################
#
# Group Info:
# agoel5 Anshuman Goel
# kgondha Kaustubh Gondhalekar
# ndas Neha Das
#
##############################################################################

p2_mpi: p2_mpi.c
	mpicc -O3 -lm -o p2_mpi p2_mpi.c
