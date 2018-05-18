# Matmul Matrix Multiplication with OpenACC
#
# Author: Dr. Richard A. Goodrum, Ph.D.
# Modified by: Jimmy Nguyen
#
# Description:
# Matrix multiplication is performed on a predetermined LxM and MxN matrix.
# OpenACC is utilized to demonstrate how parallel processing can be leveraged
# for improved performance.


# Test trials
TRIAL1  = 200 4 100
TRIAL2  = 200 10 100
TRIAL3  = 200 50 100
TRIAL4  = 200 100 100
TRIAL5  = 200 500 100
TRIAL6  = 200 1000 100
TRIAL7  = 200 2000 100
TRIAL8  = 200 5000 100
TRIAL9  = 200 10000 100
TRIAL10 = 200 20000 100

# Executables
SERIAL_EXE = matmulSerial
ACC_EXE = matmulACC
MP_EXE = matmulMP

# Make commands
$(ACC_EXE): matmul.c
	gcc -fopenacc -fopenacc-dim=1024:1:128 -o $@ $<

$(SERIAL_EXE): matmul.c
	gcc -o $@ $<

$(MP_EXE): matmul_old_omp_solution.c
	gcc -fopenmp -o $@ $<

clean:
	rm *.o matmulACC matmulSerial

run_ac: $(SERIAL_EXE) $(ACC_EXE) print_matmul serial_runs acc_runs

run_mp: $(SERIAL_EXE) $(MP_EXE)  print_matmul serial_runs openmp_runs

print_matmul:
	cat matmul.c | grep "// Matmul //" -A 30 >> results.txt

serial_runs:
	echo "Serial Runs" >> results.txt
	$(SERIAL_EXE) $(TRIAL1) >> results.txt 
	$(SERIAL_EXE) $(TRIAL2) >> results.txt 
	$(SERIAL_EXE) $(TRIAL3) >> results.txt 
	$(SERIAL_EXE) $(TRIAL4) >> results.txt 
	$(SERIAL_EXE) $(TRIAL5) >> results.txt 
	$(SERIAL_EXE) $(TRIAL6) >> results.txt 
	$(SERIAL_EXE) $(TRIAL7) >> results.txt 
	$(SERIAL_EXE) $(TRIAL8) >> results.txt 
	$(SERIAL_EXE) $(TRIAL9) >> results.txt 
	$(SERIAL_EXE) $(TRIAL10) >> results.txt
 
acc_runs:  
	echo "OpenACC Runs" >> results.txt
	$(ACC_EXE) $(TRIAL1) >> results.txt 
	$(ACC_EXE) $(TRIAL2) >> results.txt 
	$(ACC_EXE) $(TRIAL3) >> results.txt 
	$(ACC_EXE) $(TRIAL4) >> results.txt 
	$(ACC_EXE) $(TRIAL5) >> results.txt 
	$(ACC_EXE) $(TRIAL6) >> results.txt 
	$(ACC_EXE) $(TRIAL7) >> results.txt 
	$(ACC_EXE) $(TRIAL8) >> results.txt 
	$(ACC_EXE) $(TRIAL9) >> results.txt 
	$(ACC_EXE) $(TRIAL10) >> results.txt 

openmp_runs:
	echo "OpenMP Runs" >> results.txt
	$(MP_EXE) $(TRIAL1) >> results.txt 
	$(MP_EXE) $(TRIAL2) >> results.txt 
	$(MP_EXE) $(TRIAL3) >> results.txt 
	$(MP_EXE) $(TRIAL4) >> results.txt 
	$(MP_EXE) $(TRIAL5) >> results.txt 
	$(MP_EXE) $(TRIAL6) >> results.txt 
	$(MP_EXE) $(TRIAL7) >> results.txt 
	$(MP_EXE) $(TRIAL8) >> results.txt 
	$(MP_EXE) $(TRIAL9) >> results.txt 
	$(MP_EXE) $(TRIAL10) >> results.txt 



