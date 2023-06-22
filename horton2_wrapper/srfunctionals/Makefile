
fnames = LDA_ERF_exchange PBE_ERFGWS_correlation PBE_ERFGWS_exchange PBE_nomu_correlation \
		 PW92_ERF_correlation VWN5_ERF_correlation VWN5_nomu_correlation

all:
	$(foreach var, $(fnames), f2py -c $(var).F -m $(var);)

clean:
	rm *.pyf *.o *.so *.c
