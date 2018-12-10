import numpy as np

ndelst = []
epoch_itr = inpLyr = optLyr = nrm_fac = hydLyr = 0
lrn_rate = 0.0
read_wt = 0

instructions_file = "instructions.txt"
data_input_file = "data_input_train.txt"
data_output_file = "data_output_train.txt"
weight_file = ""

f=open( instructions_file , "r")
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
    read_wt_ln = lns[7].strip( ' \n' )
    if ( int( read_wt_ln[0] ) == 1 ):
        weight_file = ( read_wt_ln.split(' ') )[1]
        read_wt = 1
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
f_in = open( data_input_file ,"r")
f_out = open( data_output_file ,"r")
for lns in f_in:
    intgs = [ ( float(x) ) for x in lns.split() ]
    train_input.append( np.multiply( 1.0/nrm_fac , intgs ) )
    no_of_input_data += 1
f_in.close()
for lns in f_out:
    intgs = [ float(x) for x in lns.split() ]
    train_output.append( intgs )
f_out.close()

def read_weights(): # used for reading weights from a file
    f_wt = open( weight_file , "r")
    lns = f_wt.readlines()
    c = 0
    wtmtx = [] # the array of the corresponding weight matrices
    for i in range(hydLyr + 1):
        wt = [] # the weights
        for j in range(0, ndelst[i + 1]):
            intgs = [(float(x)) for x in lns[c].split()]
            wt.append(np.array(intgs))
            c += 1
        wtmtx.append(np.array(wt))
        c += 2
    f_wt.close()
    return wtmtx

def create_initial_wts():
    wtmtx = []  #initial weight matrix list
    for i in range(1,len(ndelst),1):
        #creating zero-centered weights
        wtmtx.append( np.random.rand( ndelst[i], ndelst[i-1] ) - .5 * np.ones( shape=( ndelst[i], ndelst[i-1] ) ,dtype=float ) )
    return wtmtx

def sigmoid(z):
    #sigmoid function
    return 1/(1+np.exp(-z))

def sigmoidPrime(z):
    #gradient of sigmoid function
    return np.exp(-z)/((1+np.exp(-z))**2)

def forward_pass( wtmtx ,lyrs ):
    lyrs_list = [] #the layers contained in a list
    lyrs_list_no_sgm = [] #the layers before the sigmoid is applied
    lyrs_list.append(lyrs)
    lyrs_list_no_sgm.append(lyrs)
    for i in range(0 , len(ndelst)-1 ):
        lyrs_list_no_sgm.append( np.matmul( wtmtx[i] , lyrs ) )
        lyrs = sigmoid( lyrs_list_no_sgm[-1] )
        lyrs_list.append( lyrs )
    return lyrs_list , lyrs_list_no_sgm

def cost_func( final_lyr , label):
    for i in range( len( final_lyr ) ):
        final_lyr[i] = final_lyr[i] - label[i] # difference between the required labels
    err = np.linalg.norm( final_lyr ) ** 2  # taking the squares
    return final_lyr , err

def backprop( wtmtx , lyrs , lyrs_list_no_sgm ):
    lyr_grad = [] # gradient for the corresponding layers
    wt_grad = [] # gradient for the weight matrices
    opt_lyr = np.multiply( 2 , lyrs[-1] ) # gradient from the error function
    x=sigmoidPrime( np.array( lyrs_list_no_sgm[-1] ) ) # gradient while passing the sigmoid layer
    opt_lyr = np.multiply( opt_lyr , x ) # final output layer gradient with weights multiplied
    lyr_grad.append(opt_lyr)
    for i in range( 2, len(lyrs)+1 ):
        x = np.matmul( lyr_grad[-1] , np.transpose( lyrs[-1*i] )  )
        wt_grad.append( x )
        opt_lyr = np.matmul( np.transpose( wtmtx[ 1-i ] ), lyr_grad[ -1 ] )
        opt_lyr = np.multiply( opt_lyr , sigmoidPrime( np.array( lyrs_list_no_sgm[-1*i] ) ) )
        lyr_grad.append( opt_lyr )
    wt_grad = wt_grad[::-1]  #reversing the array
    lyr_grad = lyr_grad[::-1] #reversing the array
    return wt_grad , lyr_grad

def wt_update( wtx_grad_dt_pts , wtx ): #updating the new weight matrix as per gradient
    return np.add( wtx , np.multiply( lrn_rate*(-1) , wtx_grad_dt_pts[0] ) )

def run( wtmx , k ):
    wt_grad_dt_pts  = [] #the gradient of the weights for different data points
    err_total = 0 #total error for all the data points
    for i in range( no_of_input_data ):
        sgm, no_sgm = forward_pass( wtmx , np.array( train_input[i] ).reshape(inpLyr, 1) )
        sgm[-1], err = cost_func( sgm[-1], train_output[i] )
        err_total += err # taking up for the total error
        wt_grad , lyrs_grad  = backprop( wtmx, sgm, no_sgm )
        wt_grad_dt_pts.append( wt_grad )
        if ( i!=0 ):
            wt_grad_dt_pts[0] = np.add( wt_grad_dt_pts[0] , wt_grad_dt_pts[1] ) #the zeroth element is the sum
            wt_grad_dt_pts = wt_grad_dt_pts[:1] #discarding the next element, the grad weight for that data point

    print( "The error for the epoch "+str(k) + " " + str(err_total) )
    return wt_update( wt_grad_dt_pts , wtmx ) , err_total

def execute():
    print( " ")
    global lrn_rate
    global read_wt
    if ( read_wt == 1 ):
        wtmx = read_weights()
    else:
        wtmx = create_initial_wts()
    wtmx , err_prev = run( wtmx, 1) #performing first iteration
    wtmx_min_err = np.copy( wtmx )
    for i in range(1 , epoch_itr):
        wtmx , err_total = run( wtmx , i+1)
        if ( err_total < err_prev ): # taking the weight matrix for minimum error
            wtmx_min_err = np.copy( wtmx )
            err_prev = err_total
    print("\n The minimum error is "+str( err_prev ))
    return wtmx_min_err

def predict( wtmx , input_lyr , result ):
    for i in range(0 , len(ndelst)-1 ):
        input_lyr = sigmoid( np.matmul( wtmx[i] , input_lyr ) )
    max_index = 0
    result = np.argmax( result )  #taking maximum no as the label
    for i in range(1 , len(input_lyr)):
        if ( input_lyr[i] > input_lyr[max_index] ):
            max_index = i
    if ( max_index == result ):
        return True
    else:
        return False


wtmtx = execute()

crct_cnt = 0
print("\nTrain set prediction ")
for i  in range( no_of_input_data ):
    if ( predict( wtmtx , train_input[i] , train_output[i]) ):
        crct_cnt += 1
print( "\n No of correct predictions is "+str(crct_cnt) )

f=open("weightMatrix.txt","w")
for i in range(len(wtmtx)):
    for j in range(len(wtmtx[i])):
        for k in range(len(wtmtx[i][j])):
            f.write( str(wtmtx[i][j][k]) +" " )
        f.write("\n")
    f.write("\n\n")