import numpy as np

ndelst = []
epoch_itr = inpLyr = optLyr = nrm_fac = hydLyr = 0
lrn_rate = 0.0

f=open("instructions.txt" , "r")
lns = f.readlines()
try:
    lrn_rate = float( lns[0].strip( ' \n' ) ) #first line should be learning rate
    epoch_itr = int( lns[1].strip( ' \n' ) ) #second line should contain no of iterations
    inpLyr = int( lns[2].strip( ' \n' ) ) #third line should contain no of nodes in input layer
    optLyr = int( lns[3].strip( ' \n' ) ) #fourth line should contain no of nodes in output layer
    nrm_fac = float( lns[4].strip( ' \n' ) ) #fifth line should contain normalization factor
    hydLyr = int( lns[5].strip( ' \n' ) ) #sixth line should contain no of hidden layer
    ndelst.append(inpLyr)
    ndelst.extend( [ int(x) for x in lns[6].strip( ' \n' ).split(' ')] ) #seventh line should contain no of nodes in hidden layer
    ndelst.append(optLyr)
    f.close()
except:
    print("Wrong Instruction list ..   Exitting code")
    exit(1)

print(" Learn Rate = "+str(lrn_rate) )
print(" No of epoch iterations = "+str(epoch_itr) )
print(" No of input layer node = "+str(inpLyr) )
print(" No of output layer node = "+str(optLyr) )
print(" No of normalization  = "+str(nrm_fac) )
print(" No of Hidden layers = "+str(hydLyr) )
print(" No of nodes in the hidden layers = " , end="")
for i in range(1,len( ndelst) - 1 ):
    print( str(ndelst[i]) , end=" ")
print("")


train_input = []
train_output = []
no_of_input_data = 0

#accepting input in the specified format and also the output
f_in = open("data_input_test.txt","r")
f_out = open("data_output_test.txt","r")
for lns in f_in:
    intgs = [ ( float(x) ) for x in lns.split() ]
    train_input.append( np.multiply( 1.0/nrm_fac , intgs ) )
    no_of_input_data += 1
f_in.close()
for lns in f_out:
    intgs = [ float(x) for x in lns.split() ]
    train_output.append( intgs )
f_out.close()

#reading the weights
f_wt = open("weightMatrix.txt","r")
lns = f_wt.readlines()
c=0
wtmtx = []
for i in range(hydLyr+1):
    wt = []
    for j in range( 0, ndelst[i+1] ):
        intgs = [(float(x)) for x in lns[c].split()]
        wt.append(np.array(intgs))
        c+=1
    wtmtx.append(np.array(wt))
    c+=2

f_wt.close()

cnf_matrix = np.zeros( shape=(optLyr,optLyr) ,dtype=int )

def sigmoid(z):
    #sigmoid function
    return 1/(1+np.exp(-z))

f_res = open("results_test.txt","w")

def predict( wtmx , input_lyr , result ):
    for i in range(0 , len(ndelst)-1 ): #calculating the last layer
        input_lyr = sigmoid( np.matmul( wtmx[i] , input_lyr ) )
    for i in range(len(input_lyr)): #writing the last layer to a file
        f_res.write( str(input_lyr[i]) + " " )
    f_res.write("\n")
    max_index = 0
    result = np.argmax( result )  #taking maximum no as the label
    for i in range(1 , len(input_lyr)):
        if ( input_lyr[i] > input_lyr[max_index] ):
            max_index = i
    if ( max_index == result ):
        print( " Correct prediction " )
        return True
    else:
        print( " Incorrect prediction predicted " + str(max_index) + " actual "+str(result) )
        return False

crct_cnt = 0
print("\nTest set prediction ")
for i  in range( no_of_input_data ):
    print( " The prediction for " + str(i+1) , end=' ' )
    if ( predict( wtmtx , train_input[i] , train_output[i]) ):
        crct_cnt += 1
print( "\n No of correct predictions is "+str(crct_cnt) )