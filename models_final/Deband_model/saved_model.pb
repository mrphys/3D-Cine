ą%
į“
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Ö
Conv3D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"!
	dilations	list(int)	

.
Identity

input"T
output"T"	
Ttype
Ą
	MaxPool3D

input"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"
Ttype:
2

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Į
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ø
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.12v2.10.0-76-gfdfc646704c8­Š
 
$Adam/conv_block3d_5/conv3d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/conv_block3d_5/conv3d_10/bias/v

8Adam/conv_block3d_5/conv3d_10/bias/v/Read/ReadVariableOpReadVariableOp$Adam/conv_block3d_5/conv3d_10/bias/v*
_output_shapes
:*
dtype0
“
&Adam/conv_block3d_5/conv3d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&Adam/conv_block3d_5/conv3d_10/kernel/v
­
:Adam/conv_block3d_5/conv3d_10/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/conv_block3d_5/conv3d_10/kernel/v**
_output_shapes
: *
dtype0

#Adam/conv_block3d_3/conv3d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/conv_block3d_3/conv3d_7/bias/v

7Adam/conv_block3d_3/conv3d_7/bias/v/Read/ReadVariableOpReadVariableOp#Adam/conv_block3d_3/conv3d_7/bias/v*
_output_shapes
:@*
dtype0
²
%Adam/conv_block3d_3/conv3d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*6
shared_name'%Adam/conv_block3d_3/conv3d_7/kernel/v
«
9Adam/conv_block3d_3/conv3d_7/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/conv_block3d_3/conv3d_7/kernel/v**
_output_shapes
:@@*
dtype0

#Adam/conv_block3d_3/conv3d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/conv_block3d_3/conv3d_6/bias/v

7Adam/conv_block3d_3/conv3d_6/bias/v/Read/ReadVariableOpReadVariableOp#Adam/conv_block3d_3/conv3d_6/bias/v*
_output_shapes
:@*
dtype0
³
%Adam/conv_block3d_3/conv3d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0* 
shape:Ą@*6
shared_name'%Adam/conv_block3d_3/conv3d_6/kernel/v
¬
9Adam/conv_block3d_3/conv3d_6/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/conv_block3d_3/conv3d_6/kernel/v*+
_output_shapes
:Ą@*
dtype0

#Adam/conv_block3d_1/conv3d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/conv_block3d_1/conv3d_3/bias/v

7Adam/conv_block3d_1/conv3d_3/bias/v/Read/ReadVariableOpReadVariableOp#Adam/conv_block3d_1/conv3d_3/bias/v*
_output_shapes
: *
dtype0
²
%Adam/conv_block3d_1/conv3d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *6
shared_name'%Adam/conv_block3d_1/conv3d_3/kernel/v
«
9Adam/conv_block3d_1/conv3d_3/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/conv_block3d_1/conv3d_3/kernel/v**
_output_shapes
:  *
dtype0

#Adam/conv_block3d_1/conv3d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/conv_block3d_1/conv3d_2/bias/v

7Adam/conv_block3d_1/conv3d_2/bias/v/Read/ReadVariableOpReadVariableOp#Adam/conv_block3d_1/conv3d_2/bias/v*
_output_shapes
: *
dtype0
²
%Adam/conv_block3d_1/conv3d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:` *6
shared_name'%Adam/conv_block3d_1/conv3d_2/kernel/v
«
9Adam/conv_block3d_1/conv3d_2/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/conv_block3d_1/conv3d_2/kernel/v**
_output_shapes
:` *
dtype0

#Adam/conv_block3d_4/conv3d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/conv_block3d_4/conv3d_9/bias/v

7Adam/conv_block3d_4/conv3d_9/bias/v/Read/ReadVariableOpReadVariableOp#Adam/conv_block3d_4/conv3d_9/bias/v*
_output_shapes	
:*
dtype0
“
%Adam/conv_block3d_4/conv3d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*!
shape:*6
shared_name'%Adam/conv_block3d_4/conv3d_9/kernel/v
­
9Adam/conv_block3d_4/conv3d_9/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/conv_block3d_4/conv3d_9/kernel/v*,
_output_shapes
:*
dtype0

#Adam/conv_block3d_4/conv3d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/conv_block3d_4/conv3d_8/bias/v

7Adam/conv_block3d_4/conv3d_8/bias/v/Read/ReadVariableOpReadVariableOp#Adam/conv_block3d_4/conv3d_8/bias/v*
_output_shapes	
:*
dtype0
³
%Adam/conv_block3d_4/conv3d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@*6
shared_name'%Adam/conv_block3d_4/conv3d_8/kernel/v
¬
9Adam/conv_block3d_4/conv3d_8/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/conv_block3d_4/conv3d_8/kernel/v*+
_output_shapes
:@*
dtype0

#Adam/conv_block3d_2/conv3d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/conv_block3d_2/conv3d_5/bias/v

7Adam/conv_block3d_2/conv3d_5/bias/v/Read/ReadVariableOpReadVariableOp#Adam/conv_block3d_2/conv3d_5/bias/v*
_output_shapes
:@*
dtype0
²
%Adam/conv_block3d_2/conv3d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*6
shared_name'%Adam/conv_block3d_2/conv3d_5/kernel/v
«
9Adam/conv_block3d_2/conv3d_5/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/conv_block3d_2/conv3d_5/kernel/v**
_output_shapes
:@@*
dtype0

#Adam/conv_block3d_2/conv3d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/conv_block3d_2/conv3d_4/bias/v

7Adam/conv_block3d_2/conv3d_4/bias/v/Read/ReadVariableOpReadVariableOp#Adam/conv_block3d_2/conv3d_4/bias/v*
_output_shapes
:@*
dtype0
²
%Adam/conv_block3d_2/conv3d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*6
shared_name'%Adam/conv_block3d_2/conv3d_4/kernel/v
«
9Adam/conv_block3d_2/conv3d_4/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/conv_block3d_2/conv3d_4/kernel/v**
_output_shapes
: @*
dtype0

!Adam/conv_block3d/conv3d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv_block3d/conv3d_1/bias/v

5Adam/conv_block3d/conv3d_1/bias/v/Read/ReadVariableOpReadVariableOp!Adam/conv_block3d/conv3d_1/bias/v*
_output_shapes
: *
dtype0
®
#Adam/conv_block3d/conv3d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *4
shared_name%#Adam/conv_block3d/conv3d_1/kernel/v
§
7Adam/conv_block3d/conv3d_1/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/conv_block3d/conv3d_1/kernel/v**
_output_shapes
:  *
dtype0

Adam/conv_block3d/conv3d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv_block3d/conv3d/bias/v

3Adam/conv_block3d/conv3d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_block3d/conv3d/bias/v*
_output_shapes
: *
dtype0
Ŗ
!Adam/conv_block3d/conv3d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv_block3d/conv3d/kernel/v
£
5Adam/conv_block3d/conv3d/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv_block3d/conv3d/kernel/v**
_output_shapes
: *
dtype0
 
$Adam/conv_block3d_5/conv3d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/conv_block3d_5/conv3d_10/bias/m

8Adam/conv_block3d_5/conv3d_10/bias/m/Read/ReadVariableOpReadVariableOp$Adam/conv_block3d_5/conv3d_10/bias/m*
_output_shapes
:*
dtype0
“
&Adam/conv_block3d_5/conv3d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&Adam/conv_block3d_5/conv3d_10/kernel/m
­
:Adam/conv_block3d_5/conv3d_10/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/conv_block3d_5/conv3d_10/kernel/m**
_output_shapes
: *
dtype0

#Adam/conv_block3d_3/conv3d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/conv_block3d_3/conv3d_7/bias/m

7Adam/conv_block3d_3/conv3d_7/bias/m/Read/ReadVariableOpReadVariableOp#Adam/conv_block3d_3/conv3d_7/bias/m*
_output_shapes
:@*
dtype0
²
%Adam/conv_block3d_3/conv3d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*6
shared_name'%Adam/conv_block3d_3/conv3d_7/kernel/m
«
9Adam/conv_block3d_3/conv3d_7/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/conv_block3d_3/conv3d_7/kernel/m**
_output_shapes
:@@*
dtype0

#Adam/conv_block3d_3/conv3d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/conv_block3d_3/conv3d_6/bias/m

7Adam/conv_block3d_3/conv3d_6/bias/m/Read/ReadVariableOpReadVariableOp#Adam/conv_block3d_3/conv3d_6/bias/m*
_output_shapes
:@*
dtype0
³
%Adam/conv_block3d_3/conv3d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0* 
shape:Ą@*6
shared_name'%Adam/conv_block3d_3/conv3d_6/kernel/m
¬
9Adam/conv_block3d_3/conv3d_6/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/conv_block3d_3/conv3d_6/kernel/m*+
_output_shapes
:Ą@*
dtype0

#Adam/conv_block3d_1/conv3d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/conv_block3d_1/conv3d_3/bias/m

7Adam/conv_block3d_1/conv3d_3/bias/m/Read/ReadVariableOpReadVariableOp#Adam/conv_block3d_1/conv3d_3/bias/m*
_output_shapes
: *
dtype0
²
%Adam/conv_block3d_1/conv3d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *6
shared_name'%Adam/conv_block3d_1/conv3d_3/kernel/m
«
9Adam/conv_block3d_1/conv3d_3/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/conv_block3d_1/conv3d_3/kernel/m**
_output_shapes
:  *
dtype0

#Adam/conv_block3d_1/conv3d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/conv_block3d_1/conv3d_2/bias/m

7Adam/conv_block3d_1/conv3d_2/bias/m/Read/ReadVariableOpReadVariableOp#Adam/conv_block3d_1/conv3d_2/bias/m*
_output_shapes
: *
dtype0
²
%Adam/conv_block3d_1/conv3d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:` *6
shared_name'%Adam/conv_block3d_1/conv3d_2/kernel/m
«
9Adam/conv_block3d_1/conv3d_2/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/conv_block3d_1/conv3d_2/kernel/m**
_output_shapes
:` *
dtype0

#Adam/conv_block3d_4/conv3d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/conv_block3d_4/conv3d_9/bias/m

7Adam/conv_block3d_4/conv3d_9/bias/m/Read/ReadVariableOpReadVariableOp#Adam/conv_block3d_4/conv3d_9/bias/m*
_output_shapes	
:*
dtype0
“
%Adam/conv_block3d_4/conv3d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*!
shape:*6
shared_name'%Adam/conv_block3d_4/conv3d_9/kernel/m
­
9Adam/conv_block3d_4/conv3d_9/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/conv_block3d_4/conv3d_9/kernel/m*,
_output_shapes
:*
dtype0

#Adam/conv_block3d_4/conv3d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/conv_block3d_4/conv3d_8/bias/m

7Adam/conv_block3d_4/conv3d_8/bias/m/Read/ReadVariableOpReadVariableOp#Adam/conv_block3d_4/conv3d_8/bias/m*
_output_shapes	
:*
dtype0
³
%Adam/conv_block3d_4/conv3d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@*6
shared_name'%Adam/conv_block3d_4/conv3d_8/kernel/m
¬
9Adam/conv_block3d_4/conv3d_8/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/conv_block3d_4/conv3d_8/kernel/m*+
_output_shapes
:@*
dtype0

#Adam/conv_block3d_2/conv3d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/conv_block3d_2/conv3d_5/bias/m

7Adam/conv_block3d_2/conv3d_5/bias/m/Read/ReadVariableOpReadVariableOp#Adam/conv_block3d_2/conv3d_5/bias/m*
_output_shapes
:@*
dtype0
²
%Adam/conv_block3d_2/conv3d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*6
shared_name'%Adam/conv_block3d_2/conv3d_5/kernel/m
«
9Adam/conv_block3d_2/conv3d_5/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/conv_block3d_2/conv3d_5/kernel/m**
_output_shapes
:@@*
dtype0

#Adam/conv_block3d_2/conv3d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/conv_block3d_2/conv3d_4/bias/m

7Adam/conv_block3d_2/conv3d_4/bias/m/Read/ReadVariableOpReadVariableOp#Adam/conv_block3d_2/conv3d_4/bias/m*
_output_shapes
:@*
dtype0
²
%Adam/conv_block3d_2/conv3d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*6
shared_name'%Adam/conv_block3d_2/conv3d_4/kernel/m
«
9Adam/conv_block3d_2/conv3d_4/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/conv_block3d_2/conv3d_4/kernel/m**
_output_shapes
: @*
dtype0

!Adam/conv_block3d/conv3d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv_block3d/conv3d_1/bias/m

5Adam/conv_block3d/conv3d_1/bias/m/Read/ReadVariableOpReadVariableOp!Adam/conv_block3d/conv3d_1/bias/m*
_output_shapes
: *
dtype0
®
#Adam/conv_block3d/conv3d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *4
shared_name%#Adam/conv_block3d/conv3d_1/kernel/m
§
7Adam/conv_block3d/conv3d_1/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/conv_block3d/conv3d_1/kernel/m**
_output_shapes
:  *
dtype0

Adam/conv_block3d/conv3d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv_block3d/conv3d/bias/m

3Adam/conv_block3d/conv3d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_block3d/conv3d/bias/m*
_output_shapes
: *
dtype0
Ŗ
!Adam/conv_block3d/conv3d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv_block3d/conv3d/kernel/m
£
5Adam/conv_block3d/conv3d/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv_block3d/conv3d/kernel/m**
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

conv_block3d_5/conv3d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameconv_block3d_5/conv3d_10/bias

1conv_block3d_5/conv3d_10/bias/Read/ReadVariableOpReadVariableOpconv_block3d_5/conv3d_10/bias*
_output_shapes
:*
dtype0
¦
conv_block3d_5/conv3d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!conv_block3d_5/conv3d_10/kernel

3conv_block3d_5/conv3d_10/kernel/Read/ReadVariableOpReadVariableOpconv_block3d_5/conv3d_10/kernel**
_output_shapes
: *
dtype0

conv_block3d_3/conv3d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameconv_block3d_3/conv3d_7/bias

0conv_block3d_3/conv3d_7/bias/Read/ReadVariableOpReadVariableOpconv_block3d_3/conv3d_7/bias*
_output_shapes
:@*
dtype0
¤
conv_block3d_3/conv3d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*/
shared_name conv_block3d_3/conv3d_7/kernel

2conv_block3d_3/conv3d_7/kernel/Read/ReadVariableOpReadVariableOpconv_block3d_3/conv3d_7/kernel**
_output_shapes
:@@*
dtype0

conv_block3d_3/conv3d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameconv_block3d_3/conv3d_6/bias

0conv_block3d_3/conv3d_6/bias/Read/ReadVariableOpReadVariableOpconv_block3d_3/conv3d_6/bias*
_output_shapes
:@*
dtype0
„
conv_block3d_3/conv3d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:Ą@*/
shared_name conv_block3d_3/conv3d_6/kernel

2conv_block3d_3/conv3d_6/kernel/Read/ReadVariableOpReadVariableOpconv_block3d_3/conv3d_6/kernel*+
_output_shapes
:Ą@*
dtype0

conv_block3d_1/conv3d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameconv_block3d_1/conv3d_3/bias

0conv_block3d_1/conv3d_3/bias/Read/ReadVariableOpReadVariableOpconv_block3d_1/conv3d_3/bias*
_output_shapes
: *
dtype0
¤
conv_block3d_1/conv3d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  */
shared_name conv_block3d_1/conv3d_3/kernel

2conv_block3d_1/conv3d_3/kernel/Read/ReadVariableOpReadVariableOpconv_block3d_1/conv3d_3/kernel**
_output_shapes
:  *
dtype0

conv_block3d_1/conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameconv_block3d_1/conv3d_2/bias

0conv_block3d_1/conv3d_2/bias/Read/ReadVariableOpReadVariableOpconv_block3d_1/conv3d_2/bias*
_output_shapes
: *
dtype0
¤
conv_block3d_1/conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:` */
shared_name conv_block3d_1/conv3d_2/kernel

2conv_block3d_1/conv3d_2/kernel/Read/ReadVariableOpReadVariableOpconv_block3d_1/conv3d_2/kernel**
_output_shapes
:` *
dtype0

conv_block3d_4/conv3d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameconv_block3d_4/conv3d_9/bias

0conv_block3d_4/conv3d_9/bias/Read/ReadVariableOpReadVariableOpconv_block3d_4/conv3d_9/bias*
_output_shapes	
:*
dtype0
¦
conv_block3d_4/conv3d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*!
shape:*/
shared_name conv_block3d_4/conv3d_9/kernel

2conv_block3d_4/conv3d_9/kernel/Read/ReadVariableOpReadVariableOpconv_block3d_4/conv3d_9/kernel*,
_output_shapes
:*
dtype0

conv_block3d_4/conv3d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameconv_block3d_4/conv3d_8/bias

0conv_block3d_4/conv3d_8/bias/Read/ReadVariableOpReadVariableOpconv_block3d_4/conv3d_8/bias*
_output_shapes	
:*
dtype0
„
conv_block3d_4/conv3d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@*/
shared_name conv_block3d_4/conv3d_8/kernel

2conv_block3d_4/conv3d_8/kernel/Read/ReadVariableOpReadVariableOpconv_block3d_4/conv3d_8/kernel*+
_output_shapes
:@*
dtype0

conv_block3d_2/conv3d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameconv_block3d_2/conv3d_5/bias

0conv_block3d_2/conv3d_5/bias/Read/ReadVariableOpReadVariableOpconv_block3d_2/conv3d_5/bias*
_output_shapes
:@*
dtype0
¤
conv_block3d_2/conv3d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*/
shared_name conv_block3d_2/conv3d_5/kernel

2conv_block3d_2/conv3d_5/kernel/Read/ReadVariableOpReadVariableOpconv_block3d_2/conv3d_5/kernel**
_output_shapes
:@@*
dtype0

conv_block3d_2/conv3d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameconv_block3d_2/conv3d_4/bias

0conv_block3d_2/conv3d_4/bias/Read/ReadVariableOpReadVariableOpconv_block3d_2/conv3d_4/bias*
_output_shapes
:@*
dtype0
¤
conv_block3d_2/conv3d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*/
shared_name conv_block3d_2/conv3d_4/kernel

2conv_block3d_2/conv3d_4/kernel/Read/ReadVariableOpReadVariableOpconv_block3d_2/conv3d_4/kernel**
_output_shapes
: @*
dtype0

conv_block3d/conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv_block3d/conv3d_1/bias

.conv_block3d/conv3d_1/bias/Read/ReadVariableOpReadVariableOpconv_block3d/conv3d_1/bias*
_output_shapes
: *
dtype0
 
conv_block3d/conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *-
shared_nameconv_block3d/conv3d_1/kernel

0conv_block3d/conv3d_1/kernel/Read/ReadVariableOpReadVariableOpconv_block3d/conv3d_1/kernel**
_output_shapes
:  *
dtype0

conv_block3d/conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv_block3d/conv3d/bias

,conv_block3d/conv3d/bias/Read/ReadVariableOpReadVariableOpconv_block3d/conv3d/bias*
_output_shapes
: *
dtype0

conv_block3d/conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv_block3d/conv3d/kernel

.conv_block3d/conv3d/kernel/Read/ReadVariableOpReadVariableOpconv_block3d/conv3d/kernel**
_output_shapes
: *
dtype0

serving_default_input_1Placeholder*5
_output_shapes#
!:’’’’’’’’’*
dtype0**
shape!:’’’’’’’’’

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv_block3d/conv3d/kernelconv_block3d/conv3d/biasconv_block3d/conv3d_1/kernelconv_block3d/conv3d_1/biasconv_block3d_2/conv3d_4/kernelconv_block3d_2/conv3d_4/biasconv_block3d_2/conv3d_5/kernelconv_block3d_2/conv3d_5/biasconv_block3d_4/conv3d_8/kernelconv_block3d_4/conv3d_8/biasconv_block3d_4/conv3d_9/kernelconv_block3d_4/conv3d_9/biasconv_block3d_3/conv3d_6/kernelconv_block3d_3/conv3d_6/biasconv_block3d_3/conv3d_7/kernelconv_block3d_3/conv3d_7/biasconv_block3d_1/conv3d_2/kernelconv_block3d_1/conv3d_2/biasconv_block3d_1/conv3d_3/kernelconv_block3d_1/conv3d_3/biasconv_block3d_5/conv3d_10/kernelconv_block3d_5/conv3d_10/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *.
f)R'
%__inference_signature_wrapper_1042749

NoOpNoOp
Ć
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÄĀ
value¹ĀBµĀ B­Ā
Å
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
_filters
	_dwt_kwargs

_enc_blocks
_dec_blocks

_pools
_upsamps
_concats

_out_block
	optimizer

signatures*
Ŗ
0
1
2
3
4
5
6
7
8
9
10
11
12
13
 14
!15
"16
#17
$18
%19
&20
'21*
Ŗ
0
1
2
3
4
5
6
7
8
9
10
11
12
13
 14
!15
"16
#17
$18
%19
&20
'21*
* 
°
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
-trace_0
.trace_1
/trace_2
0trace_3* 
6
1trace_0
2trace_1
3trace_2
4trace_3* 
* 
* 
* 

50
61
72*

80
91*

:0
;1* 

<0
=1* 

>0
?1* 
Å
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
F_filters

G_convs

H_norms
I	_dropouts*
ü
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratemČmÉmŹmĖmĢmĶmĪmĻmŠmŃmŅmÓmŌmÕ mÖ!m×"mŲ#mŁ$mŚ%mŪ&mÜ'mŻvŽvßvąvįvāvćvävåvęvēvčvévźvė vģ!vķ"vī#vļ$vš%vń&vņ'vó*

Oserving_default* 
ZT
VARIABLE_VALUEconv_block3d/conv3d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv_block3d/conv3d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv_block3d/conv3d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv_block3d/conv3d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEconv_block3d_2/conv3d_4/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv_block3d_2/conv3d_4/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEconv_block3d_2/conv3d_5/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv_block3d_2/conv3d_5/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEconv_block3d_4/conv3d_8/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv_block3d_4/conv3d_8/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEconv_block3d_4/conv3d_9/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv_block3d_4/conv3d_9/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEconv_block3d_1/conv3d_2/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv_block3d_1/conv3d_2/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEconv_block3d_1/conv3d_3/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv_block3d_1/conv3d_3/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEconv_block3d_3/conv3d_6/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv_block3d_3/conv3d_6/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEconv_block3d_3/conv3d_7/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv_block3d_3/conv3d_7/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEconv_block3d_5/conv3d_10/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEconv_block3d_5/conv3d_10/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
* 
Z
50
61
72
83
94
:5
;6
<7
=8
>9
?10
11*

P0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Å
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
W_filters

X_convs

Y_norms
Z	_dropouts*
Å
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
a_filters

b_convs

c_norms
d	_dropouts*
Å
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
k_filters

l_convs

m_norms
n	_dropouts*
Å
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses
u_filters

v_convs

w_norms
x	_dropouts*
Č
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
_filters
_convs
_norms
	_dropouts*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses* 

”	variables
¢trainable_variables
£regularization_losses
¤	keras_api
„__call__
+¦&call_and_return_all_conditional_losses* 

&0
'1*

&0
'1*
* 

§non_trainable_variables
Ølayers
©metrics
 Ŗlayer_regularization_losses
«layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*
:
¬trace_0
­trace_1
®trace_2
Ætrace_3* 
:
°trace_0
±trace_1
²trace_2
³trace_3* 
* 

“0*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
µ	variables
¶	keras_api

·total

øcount*
 
0
1
2
3*
 
0
1
2
3*
* 

¹non_trainable_variables
ŗlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*
:
¾trace_0
ætrace_1
Ątrace_2
Įtrace_3* 
:
Ātrace_0
Ćtrace_1
Ätrace_2
Åtrace_3* 
* 

Ę0
Ē1*
* 
* 
 
0
1
2
3*
 
0
1
2
3*
* 

Čnon_trainable_variables
Élayers
Źmetrics
 Ėlayer_regularization_losses
Ģlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*
:
Ķtrace_0
Ītrace_1
Ļtrace_2
Štrace_3* 
:
Ńtrace_0
Ņtrace_1
Ótrace_2
Ōtrace_3* 
* 

Õ0
Ö1*
* 
* 
 
0
1
2
3*
 
0
1
2
3*
* 

×non_trainable_variables
Ųlayers
Łmetrics
 Ślayer_regularization_losses
Ūlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*
:
Ütrace_0
Żtrace_1
Žtrace_2
ßtrace_3* 
:
ątrace_0
įtrace_1
ātrace_2
ćtrace_3* 
* 

ä0
å1*
* 
* 
 
0
1
 2
!3*
 
0
1
 2
!3*
* 

ęnon_trainable_variables
ēlayers
čmetrics
 élayer_regularization_losses
źlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*
:
ėtrace_0
ģtrace_1
ķtrace_2
ītrace_3* 
:
ļtrace_0
štrace_1
ńtrace_2
ņtrace_3* 
* 

ó0
ō1*
* 
* 
 
"0
#1
$2
%3*
 
"0
#1
$2
%3*
* 

õnon_trainable_variables
ölayers
÷metrics
 ųlayer_regularization_losses
łlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*
:
śtrace_0
ūtrace_1
ütrace_2
żtrace_3* 
:
žtrace_0
’trace_1
trace_2
trace_3* 
* 

0
1*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

 non_trainable_variables
”layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses* 

„trace_0* 

¦trace_0* 
* 
* 
* 

§non_trainable_variables
Ølayers
©metrics
 Ŗlayer_regularization_losses
«layer_metrics
”	variables
¢trainable_variables
£regularization_losses
„__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses* 

¬trace_0* 

­trace_0* 
* 

“0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ļ
®	variables
Ætrainable_variables
°regularization_losses
±	keras_api
²__call__
+³&call_and_return_all_conditional_losses

&kernel
'bias
!“_jit_compiled_convolution_op*

·0
ø1*

µ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ę0
Ē1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ļ
µ	variables
¶trainable_variables
·regularization_losses
ø	keras_api
¹__call__
+ŗ&call_and_return_all_conditional_losses

kernel
bias
!»_jit_compiled_convolution_op*
Ļ
¼	variables
½trainable_variables
¾regularization_losses
æ	keras_api
Ą__call__
+Į&call_and_return_all_conditional_losses

kernel
bias
!Ā_jit_compiled_convolution_op*
* 

Õ0
Ö1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ļ
Ć	variables
Ätrainable_variables
Åregularization_losses
Ę	keras_api
Ē__call__
+Č&call_and_return_all_conditional_losses

kernel
bias
!É_jit_compiled_convolution_op*
Ļ
Ź	variables
Ėtrainable_variables
Ģregularization_losses
Ķ	keras_api
Ī__call__
+Ļ&call_and_return_all_conditional_losses

kernel
bias
!Š_jit_compiled_convolution_op*
* 

ä0
å1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ļ
Ń	variables
Ņtrainable_variables
Óregularization_losses
Ō	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses

kernel
bias
!×_jit_compiled_convolution_op*
Ļ
Ų	variables
Łtrainable_variables
Śregularization_losses
Ū	keras_api
Ü__call__
+Ż&call_and_return_all_conditional_losses

kernel
bias
!Ž_jit_compiled_convolution_op*
* 

ó0
ō1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ļ
ß	variables
ątrainable_variables
įregularization_losses
ā	keras_api
ć__call__
+ä&call_and_return_all_conditional_losses

kernel
bias
!å_jit_compiled_convolution_op*
Ļ
ę	variables
ētrainable_variables
čregularization_losses
é	keras_api
ź__call__
+ė&call_and_return_all_conditional_losses

 kernel
!bias
!ģ_jit_compiled_convolution_op*
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ļ
ķ	variables
ītrainable_variables
ļregularization_losses
š	keras_api
ń__call__
+ņ&call_and_return_all_conditional_losses

"kernel
#bias
!ó_jit_compiled_convolution_op*
Ļ
ō	variables
õtrainable_variables
öregularization_losses
÷	keras_api
ų__call__
+ł&call_and_return_all_conditional_losses

$kernel
%bias
!ś_jit_compiled_convolution_op*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

&0
'1*

&0
'1*
* 

ūnon_trainable_variables
ülayers
żmetrics
 žlayer_regularization_losses
’layer_metrics
®	variables
Ætrainable_variables
°regularization_losses
²__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
µ	variables
¶trainable_variables
·regularization_losses
¹__call__
+ŗ&call_and_return_all_conditional_losses
'ŗ"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¼	variables
½trainable_variables
¾regularization_losses
Ą__call__
+Į&call_and_return_all_conditional_losses
'Į"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ć	variables
Ätrainable_variables
Åregularization_losses
Ē__call__
+Č&call_and_return_all_conditional_losses
'Č"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ź	variables
Ėtrainable_variables
Ģregularization_losses
Ī__call__
+Ļ&call_and_return_all_conditional_losses
'Ļ"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
 metrics
 ”layer_regularization_losses
¢layer_metrics
Ń	variables
Ņtrainable_variables
Óregularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses*

£trace_0* 

¤trace_0* 
* 

0
1*

0
1*
* 

„non_trainable_variables
¦layers
§metrics
 Ølayer_regularization_losses
©layer_metrics
Ų	variables
Łtrainable_variables
Śregularization_losses
Ü__call__
+Ż&call_and_return_all_conditional_losses
'Ż"call_and_return_conditional_losses*

Ŗtrace_0* 

«trace_0* 
* 

0
1*

0
1*
* 

¬non_trainable_variables
­layers
®metrics
 Ælayer_regularization_losses
°layer_metrics
ß	variables
ątrainable_variables
įregularization_losses
ć__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses*

±trace_0* 

²trace_0* 
* 

 0
!1*

 0
!1*
* 

³non_trainable_variables
“layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
ę	variables
ētrainable_variables
čregularization_losses
ź__call__
+ė&call_and_return_all_conditional_losses
'ė"call_and_return_conditional_losses*

øtrace_0* 

¹trace_0* 
* 

"0
#1*

"0
#1*
* 

ŗnon_trainable_variables
»layers
¼metrics
 ½layer_regularization_losses
¾layer_metrics
ķ	variables
ītrainable_variables
ļregularization_losses
ń__call__
+ņ&call_and_return_all_conditional_losses
'ņ"call_and_return_conditional_losses*

ætrace_0* 

Ątrace_0* 
* 

$0
%1*

$0
%1*
* 

Įnon_trainable_variables
Ālayers
Ćmetrics
 Älayer_regularization_losses
Ålayer_metrics
ō	variables
õtrainable_variables
öregularization_losses
ų__call__
+ł&call_and_return_all_conditional_losses
'ł"call_and_return_conditional_losses*

Ętrace_0* 

Ētrace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
}w
VARIABLE_VALUE!Adam/conv_block3d/conv3d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv_block3d/conv3d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/conv_block3d/conv3d_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/conv_block3d/conv3d_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%Adam/conv_block3d_2/conv3d_4/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/conv_block3d_2/conv3d_4/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%Adam/conv_block3d_2/conv3d_5/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/conv_block3d_2/conv3d_5/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%Adam/conv_block3d_4/conv3d_8/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/conv_block3d_4/conv3d_8/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/conv_block3d_4/conv3d_9/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/conv_block3d_4/conv3d_9/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/conv_block3d_1/conv3d_2/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/conv_block3d_1/conv3d_2/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/conv_block3d_1/conv3d_3/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/conv_block3d_1/conv3d_3/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/conv_block3d_3/conv3d_6/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/conv_block3d_3/conv3d_6/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/conv_block3d_3/conv3d_7/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/conv_block3d_3/conv3d_7/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE&Adam/conv_block3d_5/conv3d_10/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE$Adam/conv_block3d_5/conv3d_10/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/conv_block3d/conv3d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv_block3d/conv3d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/conv_block3d/conv3d_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/conv_block3d/conv3d_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%Adam/conv_block3d_2/conv3d_4/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/conv_block3d_2/conv3d_4/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%Adam/conv_block3d_2/conv3d_5/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/conv_block3d_2/conv3d_5/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%Adam/conv_block3d_4/conv3d_8/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/conv_block3d_4/conv3d_8/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/conv_block3d_4/conv3d_9/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/conv_block3d_4/conv3d_9/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/conv_block3d_1/conv3d_2/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/conv_block3d_1/conv3d_2/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/conv_block3d_1/conv3d_3/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/conv_block3d_1/conv3d_3/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/conv_block3d_3/conv3d_6/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/conv_block3d_3/conv3d_6/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/conv_block3d_3/conv3d_7/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/conv_block3d_3/conv3d_7/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE&Adam/conv_block3d_5/conv3d_10/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE$Adam/conv_block3d_5/conv3d_10/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
!
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.conv_block3d/conv3d/kernel/Read/ReadVariableOp,conv_block3d/conv3d/bias/Read/ReadVariableOp0conv_block3d/conv3d_1/kernel/Read/ReadVariableOp.conv_block3d/conv3d_1/bias/Read/ReadVariableOp2conv_block3d_2/conv3d_4/kernel/Read/ReadVariableOp0conv_block3d_2/conv3d_4/bias/Read/ReadVariableOp2conv_block3d_2/conv3d_5/kernel/Read/ReadVariableOp0conv_block3d_2/conv3d_5/bias/Read/ReadVariableOp2conv_block3d_4/conv3d_8/kernel/Read/ReadVariableOp0conv_block3d_4/conv3d_8/bias/Read/ReadVariableOp2conv_block3d_4/conv3d_9/kernel/Read/ReadVariableOp0conv_block3d_4/conv3d_9/bias/Read/ReadVariableOp2conv_block3d_1/conv3d_2/kernel/Read/ReadVariableOp0conv_block3d_1/conv3d_2/bias/Read/ReadVariableOp2conv_block3d_1/conv3d_3/kernel/Read/ReadVariableOp0conv_block3d_1/conv3d_3/bias/Read/ReadVariableOp2conv_block3d_3/conv3d_6/kernel/Read/ReadVariableOp0conv_block3d_3/conv3d_6/bias/Read/ReadVariableOp2conv_block3d_3/conv3d_7/kernel/Read/ReadVariableOp0conv_block3d_3/conv3d_7/bias/Read/ReadVariableOp3conv_block3d_5/conv3d_10/kernel/Read/ReadVariableOp1conv_block3d_5/conv3d_10/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp5Adam/conv_block3d/conv3d/kernel/m/Read/ReadVariableOp3Adam/conv_block3d/conv3d/bias/m/Read/ReadVariableOp7Adam/conv_block3d/conv3d_1/kernel/m/Read/ReadVariableOp5Adam/conv_block3d/conv3d_1/bias/m/Read/ReadVariableOp9Adam/conv_block3d_2/conv3d_4/kernel/m/Read/ReadVariableOp7Adam/conv_block3d_2/conv3d_4/bias/m/Read/ReadVariableOp9Adam/conv_block3d_2/conv3d_5/kernel/m/Read/ReadVariableOp7Adam/conv_block3d_2/conv3d_5/bias/m/Read/ReadVariableOp9Adam/conv_block3d_4/conv3d_8/kernel/m/Read/ReadVariableOp7Adam/conv_block3d_4/conv3d_8/bias/m/Read/ReadVariableOp9Adam/conv_block3d_4/conv3d_9/kernel/m/Read/ReadVariableOp7Adam/conv_block3d_4/conv3d_9/bias/m/Read/ReadVariableOp9Adam/conv_block3d_1/conv3d_2/kernel/m/Read/ReadVariableOp7Adam/conv_block3d_1/conv3d_2/bias/m/Read/ReadVariableOp9Adam/conv_block3d_1/conv3d_3/kernel/m/Read/ReadVariableOp7Adam/conv_block3d_1/conv3d_3/bias/m/Read/ReadVariableOp9Adam/conv_block3d_3/conv3d_6/kernel/m/Read/ReadVariableOp7Adam/conv_block3d_3/conv3d_6/bias/m/Read/ReadVariableOp9Adam/conv_block3d_3/conv3d_7/kernel/m/Read/ReadVariableOp7Adam/conv_block3d_3/conv3d_7/bias/m/Read/ReadVariableOp:Adam/conv_block3d_5/conv3d_10/kernel/m/Read/ReadVariableOp8Adam/conv_block3d_5/conv3d_10/bias/m/Read/ReadVariableOp5Adam/conv_block3d/conv3d/kernel/v/Read/ReadVariableOp3Adam/conv_block3d/conv3d/bias/v/Read/ReadVariableOp7Adam/conv_block3d/conv3d_1/kernel/v/Read/ReadVariableOp5Adam/conv_block3d/conv3d_1/bias/v/Read/ReadVariableOp9Adam/conv_block3d_2/conv3d_4/kernel/v/Read/ReadVariableOp7Adam/conv_block3d_2/conv3d_4/bias/v/Read/ReadVariableOp9Adam/conv_block3d_2/conv3d_5/kernel/v/Read/ReadVariableOp7Adam/conv_block3d_2/conv3d_5/bias/v/Read/ReadVariableOp9Adam/conv_block3d_4/conv3d_8/kernel/v/Read/ReadVariableOp7Adam/conv_block3d_4/conv3d_8/bias/v/Read/ReadVariableOp9Adam/conv_block3d_4/conv3d_9/kernel/v/Read/ReadVariableOp7Adam/conv_block3d_4/conv3d_9/bias/v/Read/ReadVariableOp9Adam/conv_block3d_1/conv3d_2/kernel/v/Read/ReadVariableOp7Adam/conv_block3d_1/conv3d_2/bias/v/Read/ReadVariableOp9Adam/conv_block3d_1/conv3d_3/kernel/v/Read/ReadVariableOp7Adam/conv_block3d_1/conv3d_3/bias/v/Read/ReadVariableOp9Adam/conv_block3d_3/conv3d_6/kernel/v/Read/ReadVariableOp7Adam/conv_block3d_3/conv3d_6/bias/v/Read/ReadVariableOp9Adam/conv_block3d_3/conv3d_7/kernel/v/Read/ReadVariableOp7Adam/conv_block3d_3/conv3d_7/bias/v/Read/ReadVariableOp:Adam/conv_block3d_5/conv3d_10/kernel/v/Read/ReadVariableOp8Adam/conv_block3d_5/conv3d_10/bias/v/Read/ReadVariableOpConst*V
TinO
M2K	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *)
f$R"
 __inference__traced_save_1044845
Ģ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_block3d/conv3d/kernelconv_block3d/conv3d/biasconv_block3d/conv3d_1/kernelconv_block3d/conv3d_1/biasconv_block3d_2/conv3d_4/kernelconv_block3d_2/conv3d_4/biasconv_block3d_2/conv3d_5/kernelconv_block3d_2/conv3d_5/biasconv_block3d_4/conv3d_8/kernelconv_block3d_4/conv3d_8/biasconv_block3d_4/conv3d_9/kernelconv_block3d_4/conv3d_9/biasconv_block3d_1/conv3d_2/kernelconv_block3d_1/conv3d_2/biasconv_block3d_1/conv3d_3/kernelconv_block3d_1/conv3d_3/biasconv_block3d_3/conv3d_6/kernelconv_block3d_3/conv3d_6/biasconv_block3d_3/conv3d_7/kernelconv_block3d_3/conv3d_7/biasconv_block3d_5/conv3d_10/kernelconv_block3d_5/conv3d_10/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount!Adam/conv_block3d/conv3d/kernel/mAdam/conv_block3d/conv3d/bias/m#Adam/conv_block3d/conv3d_1/kernel/m!Adam/conv_block3d/conv3d_1/bias/m%Adam/conv_block3d_2/conv3d_4/kernel/m#Adam/conv_block3d_2/conv3d_4/bias/m%Adam/conv_block3d_2/conv3d_5/kernel/m#Adam/conv_block3d_2/conv3d_5/bias/m%Adam/conv_block3d_4/conv3d_8/kernel/m#Adam/conv_block3d_4/conv3d_8/bias/m%Adam/conv_block3d_4/conv3d_9/kernel/m#Adam/conv_block3d_4/conv3d_9/bias/m%Adam/conv_block3d_1/conv3d_2/kernel/m#Adam/conv_block3d_1/conv3d_2/bias/m%Adam/conv_block3d_1/conv3d_3/kernel/m#Adam/conv_block3d_1/conv3d_3/bias/m%Adam/conv_block3d_3/conv3d_6/kernel/m#Adam/conv_block3d_3/conv3d_6/bias/m%Adam/conv_block3d_3/conv3d_7/kernel/m#Adam/conv_block3d_3/conv3d_7/bias/m&Adam/conv_block3d_5/conv3d_10/kernel/m$Adam/conv_block3d_5/conv3d_10/bias/m!Adam/conv_block3d/conv3d/kernel/vAdam/conv_block3d/conv3d/bias/v#Adam/conv_block3d/conv3d_1/kernel/v!Adam/conv_block3d/conv3d_1/bias/v%Adam/conv_block3d_2/conv3d_4/kernel/v#Adam/conv_block3d_2/conv3d_4/bias/v%Adam/conv_block3d_2/conv3d_5/kernel/v#Adam/conv_block3d_2/conv3d_5/bias/v%Adam/conv_block3d_4/conv3d_8/kernel/v#Adam/conv_block3d_4/conv3d_8/bias/v%Adam/conv_block3d_4/conv3d_9/kernel/v#Adam/conv_block3d_4/conv3d_9/bias/v%Adam/conv_block3d_1/conv3d_2/kernel/v#Adam/conv_block3d_1/conv3d_2/bias/v%Adam/conv_block3d_1/conv3d_3/kernel/v#Adam/conv_block3d_1/conv3d_3/bias/v%Adam/conv_block3d_3/conv3d_6/kernel/v#Adam/conv_block3d_3/conv3d_6/bias/v%Adam/conv_block3d_3/conv3d_7/kernel/v#Adam/conv_block3d_3/conv3d_7/bias/v&Adam/conv_block3d_5/conv3d_10/kernel/v$Adam/conv_block3d_5/conv3d_10/bias/v*U
TinN
L2J*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *,
f'R%
#__inference__traced_restore_1045074÷
ł
ė
0__inference_conv_block3d_1_layer_call_fn_1043911

inputs%
unknown:` 
	unknown_0: '
	unknown_1:  
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1041546}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’`
 
_user_specified_nameinputs
č
ó
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1043867

inputsF
'conv3d_8_conv3d_readvariableop_resource:@7
(conv3d_8_biasadd_readvariableop_resource:	G
'conv3d_9_conv3d_readvariableop_resource:7
(conv3d_9_biasadd_readvariableop_resource:	
identity¢conv3d_8/BiasAdd/ReadVariableOp¢conv3d_8/Conv3D/ReadVariableOp¢conv3d_9/BiasAdd/ReadVariableOp¢conv3d_9/Conv3D/ReadVariableOp
conv3d_8/Conv3D/ReadVariableOpReadVariableOp'conv3d_8_conv3d_readvariableop_resource*+
_output_shapes
:@*
dtype0±
conv3d_8/Conv3DConv3Dinputs&conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ *
paddingSAME*
strides	

conv3d_8/BiasAdd/ReadVariableOpReadVariableOp(conv3d_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv3d_8/BiasAddBiasAddconv3d_8/Conv3D:output:0'conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ f
ReluReluconv3d_8/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ 
conv3d_9/Conv3D/ReadVariableOpReadVariableOp'conv3d_9_conv3d_readvariableop_resource*,
_output_shapes
:*
dtype0½
conv3d_9/Conv3DConv3DRelu:activations:0&conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ *
paddingSAME*
strides	

conv3d_9/BiasAdd/ReadVariableOpReadVariableOp(conv3d_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv3d_9/BiasAddBiasAddconv3d_9/Conv3D:output:0'conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ h
Relu_1Reluconv3d_9/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ p
IdentityIdentityRelu_1:activations:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’@ Ģ
NoOpNoOp ^conv3d_8/BiasAdd/ReadVariableOp^conv3d_8/Conv3D/ReadVariableOp ^conv3d_9/BiasAdd/ReadVariableOp^conv3d_9/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’@ @: : : : 2B
conv3d_8/BiasAdd/ReadVariableOpconv3d_8/BiasAdd/ReadVariableOp2@
conv3d_8/Conv3D/ReadVariableOpconv3d_8/Conv3D/ReadVariableOp2B
conv3d_9/BiasAdd/ReadVariableOpconv3d_9/BiasAdd/ReadVariableOp2@
conv3d_9/Conv3D/ReadVariableOpconv3d_9/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:’’’’’’’’’@ @
 
_user_specified_nameinputs
Õ
Õ
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1041386

inputs/
conv3d_8_1041373:@
conv3d_8_1041375:	0
conv3d_9_1041379:
conv3d_9_1041381:	
identity¢ conv3d_8/StatefulPartitionedCall¢ conv3d_9/StatefulPartitionedCall
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_8_1041373conv3d_8_1041375*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_8_layer_call_and_return_conditional_losses_1041299v
ReluRelu)conv3d_8/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ 
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_9_1041379conv3d_9_1041381*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_9_layer_call_and_return_conditional_losses_1041316x
Relu_1Relu)conv3d_9/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ p
IdentityIdentityRelu_1:activations:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’@ 
NoOpNoOp!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’@ @: : : : 2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall:[ W
3
_output_shapes!
:’’’’’’’’’@ @
 
_user_specified_nameinputs

t
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1042040

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’
@Ąe
IdentityIdentityconcat:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’
@Ą"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:’’’’’’’’’
@:’’’’’’’’’
@@:] Y
5
_output_shapes#
!:’’’’’’’’’
@
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :’’’’’’’’’
@@
 
_user_specified_nameinputs
Ń


E__inference_conv3d_3_layer_call_and_return_conditional_losses_1041476

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0 
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’ 
 
_user_specified_nameinputs

£
*__inference_conv3d_3_layer_call_fn_1044555

inputs%
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallķ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_3_layer_call_and_return_conditional_losses_1041476}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’ 
 
_user_specified_nameinputs
é
ī
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1043929

inputsE
'conv3d_2_conv3d_readvariableop_resource:` 6
(conv3d_2_biasadd_readvariableop_resource: E
'conv3d_3_conv3d_readvariableop_resource:  6
(conv3d_3_biasadd_readvariableop_resource: 
identity¢conv3d_2/BiasAdd/ReadVariableOp¢conv3d_2/Conv3D/ReadVariableOp¢conv3d_3/BiasAdd/ReadVariableOp¢conv3d_3/Conv3D/ReadVariableOp
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:` *
dtype0²
conv3d_2/Conv3DConv3Dinputs&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	

conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ g
ReluReluconv3d_2/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0¾
conv3d_3/Conv3DConv3DRelu:activations:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	

conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ i
Relu_1Reluconv3d_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ q
IdentityIdentityRelu_1:activations:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ Ģ
NoOpNoOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’`: : : : 2B
conv3d_2/BiasAdd/ReadVariableOpconv3d_2/BiasAdd/ReadVariableOp2@
conv3d_2/Conv3D/ReadVariableOpconv3d_2/Conv3D/ReadVariableOp2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’`
 
_user_specified_nameinputs
Š
æ
D__inference_u_net3d_layer_call_and_return_conditional_losses_1043661

inputsP
2conv_block3d_conv3d_conv3d_readvariableop_resource: A
3conv_block3d_conv3d_biasadd_readvariableop_resource: R
4conv_block3d_conv3d_1_conv3d_readvariableop_resource:  C
5conv_block3d_conv3d_1_biasadd_readvariableop_resource: T
6conv_block3d_2_conv3d_4_conv3d_readvariableop_resource: @E
7conv_block3d_2_conv3d_4_biasadd_readvariableop_resource:@T
6conv_block3d_2_conv3d_5_conv3d_readvariableop_resource:@@E
7conv_block3d_2_conv3d_5_biasadd_readvariableop_resource:@U
6conv_block3d_4_conv3d_8_conv3d_readvariableop_resource:@F
7conv_block3d_4_conv3d_8_biasadd_readvariableop_resource:	V
6conv_block3d_4_conv3d_9_conv3d_readvariableop_resource:F
7conv_block3d_4_conv3d_9_biasadd_readvariableop_resource:	U
6conv_block3d_3_conv3d_6_conv3d_readvariableop_resource:Ą@E
7conv_block3d_3_conv3d_6_biasadd_readvariableop_resource:@T
6conv_block3d_3_conv3d_7_conv3d_readvariableop_resource:@@E
7conv_block3d_3_conv3d_7_biasadd_readvariableop_resource:@T
6conv_block3d_1_conv3d_2_conv3d_readvariableop_resource:` E
7conv_block3d_1_conv3d_2_biasadd_readvariableop_resource: T
6conv_block3d_1_conv3d_3_conv3d_readvariableop_resource:  E
7conv_block3d_1_conv3d_3_biasadd_readvariableop_resource: U
7conv_block3d_5_conv3d_10_conv3d_readvariableop_resource: F
8conv_block3d_5_conv3d_10_biasadd_readvariableop_resource:
identity¢*conv_block3d/conv3d/BiasAdd/ReadVariableOp¢)conv_block3d/conv3d/Conv3D/ReadVariableOp¢,conv_block3d/conv3d_1/BiasAdd/ReadVariableOp¢+conv_block3d/conv3d_1/Conv3D/ReadVariableOp¢.conv_block3d_1/conv3d_2/BiasAdd/ReadVariableOp¢-conv_block3d_1/conv3d_2/Conv3D/ReadVariableOp¢.conv_block3d_1/conv3d_3/BiasAdd/ReadVariableOp¢-conv_block3d_1/conv3d_3/Conv3D/ReadVariableOp¢.conv_block3d_2/conv3d_4/BiasAdd/ReadVariableOp¢-conv_block3d_2/conv3d_4/Conv3D/ReadVariableOp¢.conv_block3d_2/conv3d_5/BiasAdd/ReadVariableOp¢-conv_block3d_2/conv3d_5/Conv3D/ReadVariableOp¢.conv_block3d_3/conv3d_6/BiasAdd/ReadVariableOp¢-conv_block3d_3/conv3d_6/Conv3D/ReadVariableOp¢.conv_block3d_3/conv3d_7/BiasAdd/ReadVariableOp¢-conv_block3d_3/conv3d_7/Conv3D/ReadVariableOp¢.conv_block3d_4/conv3d_8/BiasAdd/ReadVariableOp¢-conv_block3d_4/conv3d_8/Conv3D/ReadVariableOp¢.conv_block3d_4/conv3d_9/BiasAdd/ReadVariableOp¢-conv_block3d_4/conv3d_9/Conv3D/ReadVariableOp¢/conv_block3d_5/conv3d_10/BiasAdd/ReadVariableOp¢.conv_block3d_5/conv3d_10/Conv3D/ReadVariableOpØ
)conv_block3d/conv3d/Conv3D/ReadVariableOpReadVariableOp2conv_block3d_conv3d_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0Č
conv_block3d/conv3d/Conv3DConv3Dinputs1conv_block3d/conv3d/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	

*conv_block3d/conv3d/BiasAdd/ReadVariableOpReadVariableOp3conv_block3d_conv3d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0æ
conv_block3d/conv3d/BiasAddBiasAdd#conv_block3d/conv3d/Conv3D:output:02conv_block3d/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
conv_block3d/ReluRelu$conv_block3d/conv3d/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ ¬
+conv_block3d/conv3d_1/Conv3D/ReadVariableOpReadVariableOp4conv_block3d_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0å
conv_block3d/conv3d_1/Conv3DConv3Dconv_block3d/Relu:activations:03conv_block3d/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	

,conv_block3d/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp5conv_block3d_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Å
conv_block3d/conv3d_1/BiasAddBiasAdd%conv_block3d/conv3d_1/Conv3D:output:04conv_block3d/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
conv_block3d/Relu_1Relu&conv_block3d/conv3d_1/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ Ć
max_pooling3d/MaxPool3D	MaxPool3D!conv_block3d/Relu_1:activations:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@ *
ksize	
*
paddingSAME*
strides	
°
-conv_block3d_2/conv3d_4/Conv3D/ReadVariableOpReadVariableOp6conv_block3d_2_conv3d_4_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0é
conv_block3d_2/conv3d_4/Conv3DConv3D max_pooling3d/MaxPool3D:output:05conv_block3d_2/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
¢
.conv_block3d_2/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp7conv_block3d_2_conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ź
conv_block3d_2/conv3d_4/BiasAddBiasAdd'conv_block3d_2/conv3d_4/Conv3D:output:06conv_block3d_2/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
conv_block3d_2/ReluRelu(conv_block3d_2/conv3d_4/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@°
-conv_block3d_2/conv3d_5/Conv3D/ReadVariableOpReadVariableOp6conv_block3d_2_conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0ź
conv_block3d_2/conv3d_5/Conv3DConv3D!conv_block3d_2/Relu:activations:05conv_block3d_2/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
¢
.conv_block3d_2/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp7conv_block3d_2_conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ź
conv_block3d_2/conv3d_5/BiasAddBiasAdd'conv_block3d_2/conv3d_5/Conv3D:output:06conv_block3d_2/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
conv_block3d_2/Relu_1Relu(conv_block3d_2/conv3d_5/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@Ę
max_pooling3d_1/MaxPool3D	MaxPool3D#conv_block3d_2/Relu_1:activations:0*
T0*3
_output_shapes!
:’’’’’’’’’@ @*
ksize	
*
paddingSAME*
strides	
±
-conv_block3d_4/conv3d_8/Conv3D/ReadVariableOpReadVariableOp6conv_block3d_4_conv3d_8_conv3d_readvariableop_resource*+
_output_shapes
:@*
dtype0ė
conv_block3d_4/conv3d_8/Conv3DConv3D"max_pooling3d_1/MaxPool3D:output:05conv_block3d_4/conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ *
paddingSAME*
strides	
£
.conv_block3d_4/conv3d_8/BiasAdd/ReadVariableOpReadVariableOp7conv_block3d_4_conv3d_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ź
conv_block3d_4/conv3d_8/BiasAddBiasAdd'conv_block3d_4/conv3d_8/Conv3D:output:06conv_block3d_4/conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ 
conv_block3d_4/ReluRelu(conv_block3d_4/conv3d_8/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ ²
-conv_block3d_4/conv3d_9/Conv3D/ReadVariableOpReadVariableOp6conv_block3d_4_conv3d_9_conv3d_readvariableop_resource*,
_output_shapes
:*
dtype0ź
conv_block3d_4/conv3d_9/Conv3DConv3D!conv_block3d_4/Relu:activations:05conv_block3d_4/conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ *
paddingSAME*
strides	
£
.conv_block3d_4/conv3d_9/BiasAdd/ReadVariableOpReadVariableOp7conv_block3d_4_conv3d_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ź
conv_block3d_4/conv3d_9/BiasAddBiasAdd'conv_block3d_4/conv3d_9/Conv3D:output:06conv_block3d_4/conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ 
conv_block3d_4/Relu_1Relu(conv_block3d_4/conv3d_9/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ a
up_sampling3d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ą
up_sampling3d_1/splitSplit(up_sampling3d_1/split/split_dim:output:0#conv_block3d_4/Relu_1:activations:0*
T0*¶
_output_shapes£
 :’’’’’’’’’@ :’’’’’’’’’@ :’’’’’’’’’@ :’’’’’’’’’@ :’’’’’’’’’@ *
	num_split]
up_sampling3d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Š
up_sampling3d_1/concatConcatV2up_sampling3d_1/split:output:0up_sampling3d_1/split:output:0up_sampling3d_1/split:output:1up_sampling3d_1/split:output:1up_sampling3d_1/split:output:2up_sampling3d_1/split:output:2up_sampling3d_1/split:output:3up_sampling3d_1/split:output:3up_sampling3d_1/split:output:4up_sampling3d_1/split:output:4$up_sampling3d_1/concat/axis:output:0*
N
*
T0*4
_output_shapes"
 :’’’’’’’’’
@ c
!up_sampling3d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
up_sampling3d_1/split_1Split*up_sampling3d_1/split_1/split_dim:output:0up_sampling3d_1/concat:output:0*
T0*
_output_shapes
:’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 *
	num_split@_
up_sampling3d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :$
up_sampling3d_1/concat_1ConcatV2 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:5 up_sampling3d_1/split_1:output:5 up_sampling3d_1/split_1:output:6 up_sampling3d_1/split_1:output:6 up_sampling3d_1/split_1:output:7 up_sampling3d_1/split_1:output:7 up_sampling3d_1/split_1:output:8 up_sampling3d_1/split_1:output:8 up_sampling3d_1/split_1:output:9 up_sampling3d_1/split_1:output:9!up_sampling3d_1/split_1:output:10!up_sampling3d_1/split_1:output:10!up_sampling3d_1/split_1:output:11!up_sampling3d_1/split_1:output:11!up_sampling3d_1/split_1:output:12!up_sampling3d_1/split_1:output:12!up_sampling3d_1/split_1:output:13!up_sampling3d_1/split_1:output:13!up_sampling3d_1/split_1:output:14!up_sampling3d_1/split_1:output:14!up_sampling3d_1/split_1:output:15!up_sampling3d_1/split_1:output:15!up_sampling3d_1/split_1:output:16!up_sampling3d_1/split_1:output:16!up_sampling3d_1/split_1:output:17!up_sampling3d_1/split_1:output:17!up_sampling3d_1/split_1:output:18!up_sampling3d_1/split_1:output:18!up_sampling3d_1/split_1:output:19!up_sampling3d_1/split_1:output:19!up_sampling3d_1/split_1:output:20!up_sampling3d_1/split_1:output:20!up_sampling3d_1/split_1:output:21!up_sampling3d_1/split_1:output:21!up_sampling3d_1/split_1:output:22!up_sampling3d_1/split_1:output:22!up_sampling3d_1/split_1:output:23!up_sampling3d_1/split_1:output:23!up_sampling3d_1/split_1:output:24!up_sampling3d_1/split_1:output:24!up_sampling3d_1/split_1:output:25!up_sampling3d_1/split_1:output:25!up_sampling3d_1/split_1:output:26!up_sampling3d_1/split_1:output:26!up_sampling3d_1/split_1:output:27!up_sampling3d_1/split_1:output:27!up_sampling3d_1/split_1:output:28!up_sampling3d_1/split_1:output:28!up_sampling3d_1/split_1:output:29!up_sampling3d_1/split_1:output:29!up_sampling3d_1/split_1:output:30!up_sampling3d_1/split_1:output:30!up_sampling3d_1/split_1:output:31!up_sampling3d_1/split_1:output:31!up_sampling3d_1/split_1:output:32!up_sampling3d_1/split_1:output:32!up_sampling3d_1/split_1:output:33!up_sampling3d_1/split_1:output:33!up_sampling3d_1/split_1:output:34!up_sampling3d_1/split_1:output:34!up_sampling3d_1/split_1:output:35!up_sampling3d_1/split_1:output:35!up_sampling3d_1/split_1:output:36!up_sampling3d_1/split_1:output:36!up_sampling3d_1/split_1:output:37!up_sampling3d_1/split_1:output:37!up_sampling3d_1/split_1:output:38!up_sampling3d_1/split_1:output:38!up_sampling3d_1/split_1:output:39!up_sampling3d_1/split_1:output:39!up_sampling3d_1/split_1:output:40!up_sampling3d_1/split_1:output:40!up_sampling3d_1/split_1:output:41!up_sampling3d_1/split_1:output:41!up_sampling3d_1/split_1:output:42!up_sampling3d_1/split_1:output:42!up_sampling3d_1/split_1:output:43!up_sampling3d_1/split_1:output:43!up_sampling3d_1/split_1:output:44!up_sampling3d_1/split_1:output:44!up_sampling3d_1/split_1:output:45!up_sampling3d_1/split_1:output:45!up_sampling3d_1/split_1:output:46!up_sampling3d_1/split_1:output:46!up_sampling3d_1/split_1:output:47!up_sampling3d_1/split_1:output:47!up_sampling3d_1/split_1:output:48!up_sampling3d_1/split_1:output:48!up_sampling3d_1/split_1:output:49!up_sampling3d_1/split_1:output:49!up_sampling3d_1/split_1:output:50!up_sampling3d_1/split_1:output:50!up_sampling3d_1/split_1:output:51!up_sampling3d_1/split_1:output:51!up_sampling3d_1/split_1:output:52!up_sampling3d_1/split_1:output:52!up_sampling3d_1/split_1:output:53!up_sampling3d_1/split_1:output:53!up_sampling3d_1/split_1:output:54!up_sampling3d_1/split_1:output:54!up_sampling3d_1/split_1:output:55!up_sampling3d_1/split_1:output:55!up_sampling3d_1/split_1:output:56!up_sampling3d_1/split_1:output:56!up_sampling3d_1/split_1:output:57!up_sampling3d_1/split_1:output:57!up_sampling3d_1/split_1:output:58!up_sampling3d_1/split_1:output:58!up_sampling3d_1/split_1:output:59!up_sampling3d_1/split_1:output:59!up_sampling3d_1/split_1:output:60!up_sampling3d_1/split_1:output:60!up_sampling3d_1/split_1:output:61!up_sampling3d_1/split_1:output:61!up_sampling3d_1/split_1:output:62!up_sampling3d_1/split_1:output:62!up_sampling3d_1/split_1:output:63!up_sampling3d_1/split_1:output:63&up_sampling3d_1/concat_1/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’
 c
!up_sampling3d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ā	
up_sampling3d_1/split_2Split*up_sampling3d_1/split_2/split_dim:output:0!up_sampling3d_1/concat_1:output:0*
T0*¶
_output_shapes£
 :’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
*
	num_split _
up_sampling3d_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :Į
up_sampling3d_1/concat_2ConcatV2 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:5 up_sampling3d_1/split_2:output:5 up_sampling3d_1/split_2:output:6 up_sampling3d_1/split_2:output:6 up_sampling3d_1/split_2:output:7 up_sampling3d_1/split_2:output:7 up_sampling3d_1/split_2:output:8 up_sampling3d_1/split_2:output:8 up_sampling3d_1/split_2:output:9 up_sampling3d_1/split_2:output:9!up_sampling3d_1/split_2:output:10!up_sampling3d_1/split_2:output:10!up_sampling3d_1/split_2:output:11!up_sampling3d_1/split_2:output:11!up_sampling3d_1/split_2:output:12!up_sampling3d_1/split_2:output:12!up_sampling3d_1/split_2:output:13!up_sampling3d_1/split_2:output:13!up_sampling3d_1/split_2:output:14!up_sampling3d_1/split_2:output:14!up_sampling3d_1/split_2:output:15!up_sampling3d_1/split_2:output:15!up_sampling3d_1/split_2:output:16!up_sampling3d_1/split_2:output:16!up_sampling3d_1/split_2:output:17!up_sampling3d_1/split_2:output:17!up_sampling3d_1/split_2:output:18!up_sampling3d_1/split_2:output:18!up_sampling3d_1/split_2:output:19!up_sampling3d_1/split_2:output:19!up_sampling3d_1/split_2:output:20!up_sampling3d_1/split_2:output:20!up_sampling3d_1/split_2:output:21!up_sampling3d_1/split_2:output:21!up_sampling3d_1/split_2:output:22!up_sampling3d_1/split_2:output:22!up_sampling3d_1/split_2:output:23!up_sampling3d_1/split_2:output:23!up_sampling3d_1/split_2:output:24!up_sampling3d_1/split_2:output:24!up_sampling3d_1/split_2:output:25!up_sampling3d_1/split_2:output:25!up_sampling3d_1/split_2:output:26!up_sampling3d_1/split_2:output:26!up_sampling3d_1/split_2:output:27!up_sampling3d_1/split_2:output:27!up_sampling3d_1/split_2:output:28!up_sampling3d_1/split_2:output:28!up_sampling3d_1/split_2:output:29!up_sampling3d_1/split_2:output:29!up_sampling3d_1/split_2:output:30!up_sampling3d_1/split_2:output:30!up_sampling3d_1/split_2:output:31!up_sampling3d_1/split_2:output:31&up_sampling3d_1/concat_2/axis:output:0*
N@*
T0*5
_output_shapes#
!:’’’’’’’’’
@[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Õ
concatenate_1/concatConcatV2!up_sampling3d_1/concat_2:output:0#conv_block3d_2/Relu_1:activations:0"concatenate_1/concat/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’
@Ą±
-conv_block3d_3/conv3d_6/Conv3D/ReadVariableOpReadVariableOp6conv_block3d_3_conv3d_6_conv3d_readvariableop_resource*+
_output_shapes
:Ą@*
dtype0ę
conv_block3d_3/conv3d_6/Conv3DConv3Dconcatenate_1/concat:output:05conv_block3d_3/conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
¢
.conv_block3d_3/conv3d_6/BiasAdd/ReadVariableOpReadVariableOp7conv_block3d_3_conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ź
conv_block3d_3/conv3d_6/BiasAddBiasAdd'conv_block3d_3/conv3d_6/Conv3D:output:06conv_block3d_3/conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
conv_block3d_3/ReluRelu(conv_block3d_3/conv3d_6/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@°
-conv_block3d_3/conv3d_7/Conv3D/ReadVariableOpReadVariableOp6conv_block3d_3_conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0ź
conv_block3d_3/conv3d_7/Conv3DConv3D!conv_block3d_3/Relu:activations:05conv_block3d_3/conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
¢
.conv_block3d_3/conv3d_7/BiasAdd/ReadVariableOpReadVariableOp7conv_block3d_3_conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ź
conv_block3d_3/conv3d_7/BiasAddBiasAdd'conv_block3d_3/conv3d_7/Conv3D:output:06conv_block3d_3/conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
conv_block3d_3/Relu_1Relu(conv_block3d_3/conv3d_7/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@_
up_sampling3d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ü
up_sampling3d/splitSplit&up_sampling3d/split/split_dim:output:0#conv_block3d_3/Relu_1:activations:0*
T0*Ö
_output_shapesĆ
Ą:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@*
	num_split
[
up_sampling3d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ä
up_sampling3d/concatConcatV2up_sampling3d/split:output:0up_sampling3d/split:output:0up_sampling3d/split:output:1up_sampling3d/split:output:1up_sampling3d/split:output:2up_sampling3d/split:output:2up_sampling3d/split:output:3up_sampling3d/split:output:3up_sampling3d/split:output:4up_sampling3d/split:output:4up_sampling3d/split:output:5up_sampling3d/split:output:5up_sampling3d/split:output:6up_sampling3d/split:output:6up_sampling3d/split:output:7up_sampling3d/split:output:7up_sampling3d/split:output:8up_sampling3d/split:output:8up_sampling3d/split:output:9up_sampling3d/split:output:9"up_sampling3d/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :’’’’’’’’’@@a
up_sampling3d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
up_sampling3d/split_1Split(up_sampling3d/split_1/split_dim:output:0up_sampling3d/concat:output:0*
T0*
_output_shapes
:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@*
	num_split]
up_sampling3d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :µC
up_sampling3d/concat_1ConcatV2up_sampling3d/split_1:output:0up_sampling3d/split_1:output:0up_sampling3d/split_1:output:1up_sampling3d/split_1:output:1up_sampling3d/split_1:output:2up_sampling3d/split_1:output:2up_sampling3d/split_1:output:3up_sampling3d/split_1:output:3up_sampling3d/split_1:output:4up_sampling3d/split_1:output:4up_sampling3d/split_1:output:5up_sampling3d/split_1:output:5up_sampling3d/split_1:output:6up_sampling3d/split_1:output:6up_sampling3d/split_1:output:7up_sampling3d/split_1:output:7up_sampling3d/split_1:output:8up_sampling3d/split_1:output:8up_sampling3d/split_1:output:9up_sampling3d/split_1:output:9up_sampling3d/split_1:output:10up_sampling3d/split_1:output:10up_sampling3d/split_1:output:11up_sampling3d/split_1:output:11up_sampling3d/split_1:output:12up_sampling3d/split_1:output:12up_sampling3d/split_1:output:13up_sampling3d/split_1:output:13up_sampling3d/split_1:output:14up_sampling3d/split_1:output:14up_sampling3d/split_1:output:15up_sampling3d/split_1:output:15up_sampling3d/split_1:output:16up_sampling3d/split_1:output:16up_sampling3d/split_1:output:17up_sampling3d/split_1:output:17up_sampling3d/split_1:output:18up_sampling3d/split_1:output:18up_sampling3d/split_1:output:19up_sampling3d/split_1:output:19up_sampling3d/split_1:output:20up_sampling3d/split_1:output:20up_sampling3d/split_1:output:21up_sampling3d/split_1:output:21up_sampling3d/split_1:output:22up_sampling3d/split_1:output:22up_sampling3d/split_1:output:23up_sampling3d/split_1:output:23up_sampling3d/split_1:output:24up_sampling3d/split_1:output:24up_sampling3d/split_1:output:25up_sampling3d/split_1:output:25up_sampling3d/split_1:output:26up_sampling3d/split_1:output:26up_sampling3d/split_1:output:27up_sampling3d/split_1:output:27up_sampling3d/split_1:output:28up_sampling3d/split_1:output:28up_sampling3d/split_1:output:29up_sampling3d/split_1:output:29up_sampling3d/split_1:output:30up_sampling3d/split_1:output:30up_sampling3d/split_1:output:31up_sampling3d/split_1:output:31up_sampling3d/split_1:output:32up_sampling3d/split_1:output:32up_sampling3d/split_1:output:33up_sampling3d/split_1:output:33up_sampling3d/split_1:output:34up_sampling3d/split_1:output:34up_sampling3d/split_1:output:35up_sampling3d/split_1:output:35up_sampling3d/split_1:output:36up_sampling3d/split_1:output:36up_sampling3d/split_1:output:37up_sampling3d/split_1:output:37up_sampling3d/split_1:output:38up_sampling3d/split_1:output:38up_sampling3d/split_1:output:39up_sampling3d/split_1:output:39up_sampling3d/split_1:output:40up_sampling3d/split_1:output:40up_sampling3d/split_1:output:41up_sampling3d/split_1:output:41up_sampling3d/split_1:output:42up_sampling3d/split_1:output:42up_sampling3d/split_1:output:43up_sampling3d/split_1:output:43up_sampling3d/split_1:output:44up_sampling3d/split_1:output:44up_sampling3d/split_1:output:45up_sampling3d/split_1:output:45up_sampling3d/split_1:output:46up_sampling3d/split_1:output:46up_sampling3d/split_1:output:47up_sampling3d/split_1:output:47up_sampling3d/split_1:output:48up_sampling3d/split_1:output:48up_sampling3d/split_1:output:49up_sampling3d/split_1:output:49up_sampling3d/split_1:output:50up_sampling3d/split_1:output:50up_sampling3d/split_1:output:51up_sampling3d/split_1:output:51up_sampling3d/split_1:output:52up_sampling3d/split_1:output:52up_sampling3d/split_1:output:53up_sampling3d/split_1:output:53up_sampling3d/split_1:output:54up_sampling3d/split_1:output:54up_sampling3d/split_1:output:55up_sampling3d/split_1:output:55up_sampling3d/split_1:output:56up_sampling3d/split_1:output:56up_sampling3d/split_1:output:57up_sampling3d/split_1:output:57up_sampling3d/split_1:output:58up_sampling3d/split_1:output:58up_sampling3d/split_1:output:59up_sampling3d/split_1:output:59up_sampling3d/split_1:output:60up_sampling3d/split_1:output:60up_sampling3d/split_1:output:61up_sampling3d/split_1:output:61up_sampling3d/split_1:output:62up_sampling3d/split_1:output:62up_sampling3d/split_1:output:63up_sampling3d/split_1:output:63up_sampling3d/split_1:output:64up_sampling3d/split_1:output:64up_sampling3d/split_1:output:65up_sampling3d/split_1:output:65up_sampling3d/split_1:output:66up_sampling3d/split_1:output:66up_sampling3d/split_1:output:67up_sampling3d/split_1:output:67up_sampling3d/split_1:output:68up_sampling3d/split_1:output:68up_sampling3d/split_1:output:69up_sampling3d/split_1:output:69up_sampling3d/split_1:output:70up_sampling3d/split_1:output:70up_sampling3d/split_1:output:71up_sampling3d/split_1:output:71up_sampling3d/split_1:output:72up_sampling3d/split_1:output:72up_sampling3d/split_1:output:73up_sampling3d/split_1:output:73up_sampling3d/split_1:output:74up_sampling3d/split_1:output:74up_sampling3d/split_1:output:75up_sampling3d/split_1:output:75up_sampling3d/split_1:output:76up_sampling3d/split_1:output:76up_sampling3d/split_1:output:77up_sampling3d/split_1:output:77up_sampling3d/split_1:output:78up_sampling3d/split_1:output:78up_sampling3d/split_1:output:79up_sampling3d/split_1:output:79up_sampling3d/split_1:output:80up_sampling3d/split_1:output:80up_sampling3d/split_1:output:81up_sampling3d/split_1:output:81up_sampling3d/split_1:output:82up_sampling3d/split_1:output:82up_sampling3d/split_1:output:83up_sampling3d/split_1:output:83up_sampling3d/split_1:output:84up_sampling3d/split_1:output:84up_sampling3d/split_1:output:85up_sampling3d/split_1:output:85up_sampling3d/split_1:output:86up_sampling3d/split_1:output:86up_sampling3d/split_1:output:87up_sampling3d/split_1:output:87up_sampling3d/split_1:output:88up_sampling3d/split_1:output:88up_sampling3d/split_1:output:89up_sampling3d/split_1:output:89up_sampling3d/split_1:output:90up_sampling3d/split_1:output:90up_sampling3d/split_1:output:91up_sampling3d/split_1:output:91up_sampling3d/split_1:output:92up_sampling3d/split_1:output:92up_sampling3d/split_1:output:93up_sampling3d/split_1:output:93up_sampling3d/split_1:output:94up_sampling3d/split_1:output:94up_sampling3d/split_1:output:95up_sampling3d/split_1:output:95up_sampling3d/split_1:output:96up_sampling3d/split_1:output:96up_sampling3d/split_1:output:97up_sampling3d/split_1:output:97up_sampling3d/split_1:output:98up_sampling3d/split_1:output:98up_sampling3d/split_1:output:99up_sampling3d/split_1:output:99 up_sampling3d/split_1:output:100 up_sampling3d/split_1:output:100 up_sampling3d/split_1:output:101 up_sampling3d/split_1:output:101 up_sampling3d/split_1:output:102 up_sampling3d/split_1:output:102 up_sampling3d/split_1:output:103 up_sampling3d/split_1:output:103 up_sampling3d/split_1:output:104 up_sampling3d/split_1:output:104 up_sampling3d/split_1:output:105 up_sampling3d/split_1:output:105 up_sampling3d/split_1:output:106 up_sampling3d/split_1:output:106 up_sampling3d/split_1:output:107 up_sampling3d/split_1:output:107 up_sampling3d/split_1:output:108 up_sampling3d/split_1:output:108 up_sampling3d/split_1:output:109 up_sampling3d/split_1:output:109 up_sampling3d/split_1:output:110 up_sampling3d/split_1:output:110 up_sampling3d/split_1:output:111 up_sampling3d/split_1:output:111 up_sampling3d/split_1:output:112 up_sampling3d/split_1:output:112 up_sampling3d/split_1:output:113 up_sampling3d/split_1:output:113 up_sampling3d/split_1:output:114 up_sampling3d/split_1:output:114 up_sampling3d/split_1:output:115 up_sampling3d/split_1:output:115 up_sampling3d/split_1:output:116 up_sampling3d/split_1:output:116 up_sampling3d/split_1:output:117 up_sampling3d/split_1:output:117 up_sampling3d/split_1:output:118 up_sampling3d/split_1:output:118 up_sampling3d/split_1:output:119 up_sampling3d/split_1:output:119 up_sampling3d/split_1:output:120 up_sampling3d/split_1:output:120 up_sampling3d/split_1:output:121 up_sampling3d/split_1:output:121 up_sampling3d/split_1:output:122 up_sampling3d/split_1:output:122 up_sampling3d/split_1:output:123 up_sampling3d/split_1:output:123 up_sampling3d/split_1:output:124 up_sampling3d/split_1:output:124 up_sampling3d/split_1:output:125 up_sampling3d/split_1:output:125 up_sampling3d/split_1:output:126 up_sampling3d/split_1:output:126 up_sampling3d/split_1:output:127 up_sampling3d/split_1:output:127$up_sampling3d/concat_1/axis:output:0*
N*
T0*4
_output_shapes"
 :’’’’’’’’’@@a
up_sampling3d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d/split_2Split(up_sampling3d/split_2/split_dim:output:0up_sampling3d/concat_1:output:0*
T0*
_output_shapes
:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@*
	num_split@]
up_sampling3d/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :ž!
up_sampling3d/concat_2ConcatV2up_sampling3d/split_2:output:0up_sampling3d/split_2:output:0up_sampling3d/split_2:output:1up_sampling3d/split_2:output:1up_sampling3d/split_2:output:2up_sampling3d/split_2:output:2up_sampling3d/split_2:output:3up_sampling3d/split_2:output:3up_sampling3d/split_2:output:4up_sampling3d/split_2:output:4up_sampling3d/split_2:output:5up_sampling3d/split_2:output:5up_sampling3d/split_2:output:6up_sampling3d/split_2:output:6up_sampling3d/split_2:output:7up_sampling3d/split_2:output:7up_sampling3d/split_2:output:8up_sampling3d/split_2:output:8up_sampling3d/split_2:output:9up_sampling3d/split_2:output:9up_sampling3d/split_2:output:10up_sampling3d/split_2:output:10up_sampling3d/split_2:output:11up_sampling3d/split_2:output:11up_sampling3d/split_2:output:12up_sampling3d/split_2:output:12up_sampling3d/split_2:output:13up_sampling3d/split_2:output:13up_sampling3d/split_2:output:14up_sampling3d/split_2:output:14up_sampling3d/split_2:output:15up_sampling3d/split_2:output:15up_sampling3d/split_2:output:16up_sampling3d/split_2:output:16up_sampling3d/split_2:output:17up_sampling3d/split_2:output:17up_sampling3d/split_2:output:18up_sampling3d/split_2:output:18up_sampling3d/split_2:output:19up_sampling3d/split_2:output:19up_sampling3d/split_2:output:20up_sampling3d/split_2:output:20up_sampling3d/split_2:output:21up_sampling3d/split_2:output:21up_sampling3d/split_2:output:22up_sampling3d/split_2:output:22up_sampling3d/split_2:output:23up_sampling3d/split_2:output:23up_sampling3d/split_2:output:24up_sampling3d/split_2:output:24up_sampling3d/split_2:output:25up_sampling3d/split_2:output:25up_sampling3d/split_2:output:26up_sampling3d/split_2:output:26up_sampling3d/split_2:output:27up_sampling3d/split_2:output:27up_sampling3d/split_2:output:28up_sampling3d/split_2:output:28up_sampling3d/split_2:output:29up_sampling3d/split_2:output:29up_sampling3d/split_2:output:30up_sampling3d/split_2:output:30up_sampling3d/split_2:output:31up_sampling3d/split_2:output:31up_sampling3d/split_2:output:32up_sampling3d/split_2:output:32up_sampling3d/split_2:output:33up_sampling3d/split_2:output:33up_sampling3d/split_2:output:34up_sampling3d/split_2:output:34up_sampling3d/split_2:output:35up_sampling3d/split_2:output:35up_sampling3d/split_2:output:36up_sampling3d/split_2:output:36up_sampling3d/split_2:output:37up_sampling3d/split_2:output:37up_sampling3d/split_2:output:38up_sampling3d/split_2:output:38up_sampling3d/split_2:output:39up_sampling3d/split_2:output:39up_sampling3d/split_2:output:40up_sampling3d/split_2:output:40up_sampling3d/split_2:output:41up_sampling3d/split_2:output:41up_sampling3d/split_2:output:42up_sampling3d/split_2:output:42up_sampling3d/split_2:output:43up_sampling3d/split_2:output:43up_sampling3d/split_2:output:44up_sampling3d/split_2:output:44up_sampling3d/split_2:output:45up_sampling3d/split_2:output:45up_sampling3d/split_2:output:46up_sampling3d/split_2:output:46up_sampling3d/split_2:output:47up_sampling3d/split_2:output:47up_sampling3d/split_2:output:48up_sampling3d/split_2:output:48up_sampling3d/split_2:output:49up_sampling3d/split_2:output:49up_sampling3d/split_2:output:50up_sampling3d/split_2:output:50up_sampling3d/split_2:output:51up_sampling3d/split_2:output:51up_sampling3d/split_2:output:52up_sampling3d/split_2:output:52up_sampling3d/split_2:output:53up_sampling3d/split_2:output:53up_sampling3d/split_2:output:54up_sampling3d/split_2:output:54up_sampling3d/split_2:output:55up_sampling3d/split_2:output:55up_sampling3d/split_2:output:56up_sampling3d/split_2:output:56up_sampling3d/split_2:output:57up_sampling3d/split_2:output:57up_sampling3d/split_2:output:58up_sampling3d/split_2:output:58up_sampling3d/split_2:output:59up_sampling3d/split_2:output:59up_sampling3d/split_2:output:60up_sampling3d/split_2:output:60up_sampling3d/split_2:output:61up_sampling3d/split_2:output:61up_sampling3d/split_2:output:62up_sampling3d/split_2:output:62up_sampling3d/split_2:output:63up_sampling3d/split_2:output:63$up_sampling3d/concat_2/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’@Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ķ
concatenate/concatConcatV2up_sampling3d/concat_2:output:0!conv_block3d/Relu_1:activations:0 concatenate/concat/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’`°
-conv_block3d_1/conv3d_2/Conv3D/ReadVariableOpReadVariableOp6conv_block3d_1_conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:` *
dtype0å
conv_block3d_1/conv3d_2/Conv3DConv3Dconcatenate/concat:output:05conv_block3d_1/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	
¢
.conv_block3d_1/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp7conv_block3d_1_conv3d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ė
conv_block3d_1/conv3d_2/BiasAddBiasAdd'conv_block3d_1/conv3d_2/Conv3D:output:06conv_block3d_1/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
conv_block3d_1/ReluRelu(conv_block3d_1/conv3d_2/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ °
-conv_block3d_1/conv3d_3/Conv3D/ReadVariableOpReadVariableOp6conv_block3d_1_conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0ė
conv_block3d_1/conv3d_3/Conv3DConv3D!conv_block3d_1/Relu:activations:05conv_block3d_1/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	
¢
.conv_block3d_1/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp7conv_block3d_1_conv3d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ė
conv_block3d_1/conv3d_3/BiasAddBiasAdd'conv_block3d_1/conv3d_3/Conv3D:output:06conv_block3d_1/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
conv_block3d_1/Relu_1Relu(conv_block3d_1/conv3d_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ ²
.conv_block3d_5/conv3d_10/Conv3D/ReadVariableOpReadVariableOp7conv_block3d_5_conv3d_10_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0ļ
conv_block3d_5/conv3d_10/Conv3DConv3D#conv_block3d_1/Relu_1:activations:06conv_block3d_5/conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’*
paddingSAME*
strides	
¤
/conv_block3d_5/conv3d_10/BiasAdd/ReadVariableOpReadVariableOp8conv_block3d_5_conv3d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ī
 conv_block3d_5/conv3d_10/BiasAddBiasAdd(conv_block3d_5/conv3d_10/Conv3D:output:07conv_block3d_5/conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’
IdentityIdentity)conv_block3d_5/conv3d_10/BiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ē
NoOpNoOp+^conv_block3d/conv3d/BiasAdd/ReadVariableOp*^conv_block3d/conv3d/Conv3D/ReadVariableOp-^conv_block3d/conv3d_1/BiasAdd/ReadVariableOp,^conv_block3d/conv3d_1/Conv3D/ReadVariableOp/^conv_block3d_1/conv3d_2/BiasAdd/ReadVariableOp.^conv_block3d_1/conv3d_2/Conv3D/ReadVariableOp/^conv_block3d_1/conv3d_3/BiasAdd/ReadVariableOp.^conv_block3d_1/conv3d_3/Conv3D/ReadVariableOp/^conv_block3d_2/conv3d_4/BiasAdd/ReadVariableOp.^conv_block3d_2/conv3d_4/Conv3D/ReadVariableOp/^conv_block3d_2/conv3d_5/BiasAdd/ReadVariableOp.^conv_block3d_2/conv3d_5/Conv3D/ReadVariableOp/^conv_block3d_3/conv3d_6/BiasAdd/ReadVariableOp.^conv_block3d_3/conv3d_6/Conv3D/ReadVariableOp/^conv_block3d_3/conv3d_7/BiasAdd/ReadVariableOp.^conv_block3d_3/conv3d_7/Conv3D/ReadVariableOp/^conv_block3d_4/conv3d_8/BiasAdd/ReadVariableOp.^conv_block3d_4/conv3d_8/Conv3D/ReadVariableOp/^conv_block3d_4/conv3d_9/BiasAdd/ReadVariableOp.^conv_block3d_4/conv3d_9/Conv3D/ReadVariableOp0^conv_block3d_5/conv3d_10/BiasAdd/ReadVariableOp/^conv_block3d_5/conv3d_10/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : 2X
*conv_block3d/conv3d/BiasAdd/ReadVariableOp*conv_block3d/conv3d/BiasAdd/ReadVariableOp2V
)conv_block3d/conv3d/Conv3D/ReadVariableOp)conv_block3d/conv3d/Conv3D/ReadVariableOp2\
,conv_block3d/conv3d_1/BiasAdd/ReadVariableOp,conv_block3d/conv3d_1/BiasAdd/ReadVariableOp2Z
+conv_block3d/conv3d_1/Conv3D/ReadVariableOp+conv_block3d/conv3d_1/Conv3D/ReadVariableOp2`
.conv_block3d_1/conv3d_2/BiasAdd/ReadVariableOp.conv_block3d_1/conv3d_2/BiasAdd/ReadVariableOp2^
-conv_block3d_1/conv3d_2/Conv3D/ReadVariableOp-conv_block3d_1/conv3d_2/Conv3D/ReadVariableOp2`
.conv_block3d_1/conv3d_3/BiasAdd/ReadVariableOp.conv_block3d_1/conv3d_3/BiasAdd/ReadVariableOp2^
-conv_block3d_1/conv3d_3/Conv3D/ReadVariableOp-conv_block3d_1/conv3d_3/Conv3D/ReadVariableOp2`
.conv_block3d_2/conv3d_4/BiasAdd/ReadVariableOp.conv_block3d_2/conv3d_4/BiasAdd/ReadVariableOp2^
-conv_block3d_2/conv3d_4/Conv3D/ReadVariableOp-conv_block3d_2/conv3d_4/Conv3D/ReadVariableOp2`
.conv_block3d_2/conv3d_5/BiasAdd/ReadVariableOp.conv_block3d_2/conv3d_5/BiasAdd/ReadVariableOp2^
-conv_block3d_2/conv3d_5/Conv3D/ReadVariableOp-conv_block3d_2/conv3d_5/Conv3D/ReadVariableOp2`
.conv_block3d_3/conv3d_6/BiasAdd/ReadVariableOp.conv_block3d_3/conv3d_6/BiasAdd/ReadVariableOp2^
-conv_block3d_3/conv3d_6/Conv3D/ReadVariableOp-conv_block3d_3/conv3d_6/Conv3D/ReadVariableOp2`
.conv_block3d_3/conv3d_7/BiasAdd/ReadVariableOp.conv_block3d_3/conv3d_7/BiasAdd/ReadVariableOp2^
-conv_block3d_3/conv3d_7/Conv3D/ReadVariableOp-conv_block3d_3/conv3d_7/Conv3D/ReadVariableOp2`
.conv_block3d_4/conv3d_8/BiasAdd/ReadVariableOp.conv_block3d_4/conv3d_8/BiasAdd/ReadVariableOp2^
-conv_block3d_4/conv3d_8/Conv3D/ReadVariableOp-conv_block3d_4/conv3d_8/Conv3D/ReadVariableOp2`
.conv_block3d_4/conv3d_9/BiasAdd/ReadVariableOp.conv_block3d_4/conv3d_9/BiasAdd/ReadVariableOp2^
-conv_block3d_4/conv3d_9/Conv3D/ReadVariableOp-conv_block3d_4/conv3d_9/Conv3D/ReadVariableOp2b
/conv_block3d_5/conv3d_10/BiasAdd/ReadVariableOp/conv_block3d_5/conv3d_10/BiasAdd/ReadVariableOp2`
.conv_block3d_5/conv3d_10/Conv3D/ReadVariableOp.conv_block3d_5/conv3d_10/Conv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’
 
_user_specified_nameinputs
ē
Ē
)__inference_u_net3d_layer_call_fn_1042798

inputs%
unknown: 
	unknown_0: '
	unknown_1:  
	unknown_2: '
	unknown_3: @
	unknown_4:@'
	unknown_5:@@
	unknown_6:@(
	unknown_7:@
	unknown_8:	)
	unknown_9:

unknown_10:	)

unknown_11:Ą@

unknown_12:@(

unknown_13:@@

unknown_14:@(

unknown_15:` 

unknown_16: (

unknown_17:  

unknown_18: (

unknown_19: 

unknown_20:
identity¢StatefulPartitionedCallū
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_u_net3d_layer_call_and_return_conditional_losses_1042292}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’
 
_user_specified_nameinputs
ą
ī
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1043805

inputsE
'conv3d_4_conv3d_readvariableop_resource: @6
(conv3d_4_biasadd_readvariableop_resource:@E
'conv3d_5_conv3d_readvariableop_resource:@@6
(conv3d_5_biasadd_readvariableop_resource:@
identity¢conv3d_4/BiasAdd/ReadVariableOp¢conv3d_4/Conv3D/ReadVariableOp¢conv3d_5/BiasAdd/ReadVariableOp¢conv3d_5/Conv3D/ReadVariableOp
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0±
conv3d_4/Conv3DConv3Dinputs&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	

conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@f
ReluReluconv3d_4/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0½
conv3d_5/Conv3DConv3DRelu:activations:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	

conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@h
Relu_1Reluconv3d_5/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@p
IdentityIdentityRelu_1:activations:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@Ģ
NoOpNoOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:’’’’’’’’’
@ : : : : 2B
conv3d_4/BiasAdd/ReadVariableOpconv3d_4/BiasAdd/ReadVariableOp2@
conv3d_4/Conv3D/ReadVariableOpconv3d_4/Conv3D/ReadVariableOp2B
conv3d_5/BiasAdd/ReadVariableOpconv3d_5/BiasAdd/ReadVariableOp2@
conv3d_5/Conv3D/ReadVariableOpconv3d_5/Conv3D/ReadVariableOp:\ X
4
_output_shapes"
 :’’’’’’’’’
@ 
 
_user_specified_nameinputs
Š


E__inference_conv3d_6_layer_call_and_return_conditional_losses_1044584

inputs=
conv3d_readvariableop_resource:Ą@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:Ą@*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’
@Ą: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’
@Ą
 
_user_specified_nameinputs
õ
é
.__inference_conv_block3d_layer_call_fn_1043712

inputs%
unknown: 
	unknown_0: '
	unknown_1:  
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1041004}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’
 
_user_specified_nameinputs
Ł
Š
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1041546

inputs.
conv3d_2_1041533:` 
conv3d_2_1041535: .
conv3d_3_1041539:  
conv3d_3_1041541: 
identity¢ conv3d_2/StatefulPartitionedCall¢ conv3d_3/StatefulPartitionedCall
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_2_1041533conv3d_2_1041535*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_2_layer_call_and_return_conditional_losses_1041459w
ReluRelu)conv3d_2/StatefulPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_3_1041539conv3d_3_1041541*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_3_layer_call_and_return_conditional_losses_1041476y
Relu_1Relu)conv3d_3/StatefulPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ q
IdentityIdentityRelu_1:activations:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ 
NoOpNoOp!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’`: : : : 2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’`
 
_user_specified_nameinputs
õ
ė
0__inference_conv_block3d_2_layer_call_fn_1043774

inputs%
unknown: @
	unknown_0:@'
	unknown_1:@@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1041164|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:’’’’’’’’’
@ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :’’’’’’’’’
@ 
 
_user_specified_nameinputs

£
*__inference_conv3d_1_layer_call_fn_1044441

inputs%
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallķ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_1_layer_call_and_return_conditional_losses_1040996}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’ 
 
_user_specified_nameinputs
č
ó
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1043885

inputsF
'conv3d_8_conv3d_readvariableop_resource:@7
(conv3d_8_biasadd_readvariableop_resource:	G
'conv3d_9_conv3d_readvariableop_resource:7
(conv3d_9_biasadd_readvariableop_resource:	
identity¢conv3d_8/BiasAdd/ReadVariableOp¢conv3d_8/Conv3D/ReadVariableOp¢conv3d_9/BiasAdd/ReadVariableOp¢conv3d_9/Conv3D/ReadVariableOp
conv3d_8/Conv3D/ReadVariableOpReadVariableOp'conv3d_8_conv3d_readvariableop_resource*+
_output_shapes
:@*
dtype0±
conv3d_8/Conv3DConv3Dinputs&conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ *
paddingSAME*
strides	

conv3d_8/BiasAdd/ReadVariableOpReadVariableOp(conv3d_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv3d_8/BiasAddBiasAddconv3d_8/Conv3D:output:0'conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ f
ReluReluconv3d_8/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ 
conv3d_9/Conv3D/ReadVariableOpReadVariableOp'conv3d_9_conv3d_readvariableop_resource*,
_output_shapes
:*
dtype0½
conv3d_9/Conv3DConv3DRelu:activations:0&conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ *
paddingSAME*
strides	

conv3d_9/BiasAdd/ReadVariableOpReadVariableOp(conv3d_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv3d_9/BiasAddBiasAddconv3d_9/Conv3D:output:0'conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ h
Relu_1Reluconv3d_9/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ p
IdentityIdentityRelu_1:activations:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’@ Ģ
NoOpNoOp ^conv3d_8/BiasAdd/ReadVariableOp^conv3d_8/Conv3D/ReadVariableOp ^conv3d_9/BiasAdd/ReadVariableOp^conv3d_9/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’@ @: : : : 2B
conv3d_8/BiasAdd/ReadVariableOpconv3d_8/BiasAdd/ReadVariableOp2@
conv3d_8/Conv3D/ReadVariableOpconv3d_8/Conv3D/ReadVariableOp2B
conv3d_9/BiasAdd/ReadVariableOpconv3d_9/BiasAdd/ReadVariableOp2@
conv3d_9/Conv3D/ReadVariableOpconv3d_9/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:’’’’’’’’’@ @
 
_user_specified_nameinputs
Ń


E__inference_conv3d_2_layer_call_and_return_conditional_losses_1044546

inputs<
conv3d_readvariableop_resource:` -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:` *
dtype0 
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’`
 
_user_specified_nameinputs
	
ą
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1041810

inputs/
conv3d_10_1041804: 
conv3d_10_1041806:
identity¢!conv3d_10/StatefulPartitionedCall
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_10_1041804conv3d_10_1041806*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv3d_10_layer_call_and_return_conditional_losses_1041803
IdentityIdentity*conv3d_10/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’j
NoOpNoOp"^conv3d_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’ 
 
_user_specified_nameinputs
Ä
É
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1041122
input_1,
conv3d_1041109: 
conv3d_1041111: .
conv3d_1_1041115:  
conv3d_1_1041117: 
identity¢conv3d/StatefulPartitionedCall¢ conv3d_1/StatefulPartitionedCall’
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_1041109conv3d_1041111*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_conv3d_layer_call_and_return_conditional_losses_1040979u
ReluRelu'conv3d/StatefulPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_1_1041115conv3d_1_1041117*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_1_layer_call_and_return_conditional_losses_1040996y
Relu_1Relu)conv3d_1/StatefulPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ q
IdentityIdentityRelu_1:activations:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ 
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’: : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’
!
_user_specified_name	input_1
õ
ė
0__inference_conv_block3d_2_layer_call_fn_1043787

inputs%
unknown: @
	unknown_0:@'
	unknown_1:@@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1041226|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:’’’’’’’’’
@ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :’’’’’’’’’
@ 
 
_user_specified_nameinputs
ą
ī
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1043823

inputsE
'conv3d_4_conv3d_readvariableop_resource: @6
(conv3d_4_biasadd_readvariableop_resource:@E
'conv3d_5_conv3d_readvariableop_resource:@@6
(conv3d_5_biasadd_readvariableop_resource:@
identity¢conv3d_4/BiasAdd/ReadVariableOp¢conv3d_4/Conv3D/ReadVariableOp¢conv3d_5/BiasAdd/ReadVariableOp¢conv3d_5/Conv3D/ReadVariableOp
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0±
conv3d_4/Conv3DConv3Dinputs&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	

conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@f
ReluReluconv3d_4/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0½
conv3d_5/Conv3DConv3DRelu:activations:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	

conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@h
Relu_1Reluconv3d_5/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@p
IdentityIdentityRelu_1:activations:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@Ģ
NoOpNoOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:’’’’’’’’’
@ : : : : 2B
conv3d_4/BiasAdd/ReadVariableOpconv3d_4/BiasAdd/ReadVariableOp2@
conv3d_4/Conv3D/ReadVariableOpconv3d_4/Conv3D/ReadVariableOp2B
conv3d_5/BiasAdd/ReadVariableOpconv3d_5/BiasAdd/ReadVariableOp2@
conv3d_5/Conv3D/ReadVariableOpconv3d_5/Conv3D/ReadVariableOp:\ X
4
_output_shapes"
 :’’’’’’’’’
@ 
 
_user_specified_nameinputs
ų
ź
.__inference_conv_block3d_layer_call_fn_1041090
input_1%
unknown: 
	unknown_0: '
	unknown_1:  
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1041066}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’
!
_user_specified_name	input_1

¤
*__inference_conv3d_6_layer_call_fn_1044574

inputs&
unknown:Ą@
	unknown_0:@
identity¢StatefulPartitionedCallģ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_6_layer_call_and_return_conditional_losses_1041619|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’
@Ą: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’
@Ą
 
_user_specified_nameinputs
Ė=
É

D__inference_u_net3d_layer_call_and_return_conditional_losses_1042476

inputs2
conv_block3d_1042419: "
conv_block3d_1042421: 2
conv_block3d_1042423:  "
conv_block3d_1042425: 4
conv_block3d_2_1042429: @$
conv_block3d_2_1042431:@4
conv_block3d_2_1042433:@@$
conv_block3d_2_1042435:@5
conv_block3d_4_1042439:@%
conv_block3d_4_1042441:	6
conv_block3d_4_1042443:%
conv_block3d_4_1042445:	5
conv_block3d_3_1042450:Ą@$
conv_block3d_3_1042452:@4
conv_block3d_3_1042454:@@$
conv_block3d_3_1042456:@4
conv_block3d_1_1042461:` $
conv_block3d_1_1042463: 4
conv_block3d_1_1042465:  $
conv_block3d_1_1042467: 4
conv_block3d_5_1042470: $
conv_block3d_5_1042472:
identity¢$conv_block3d/StatefulPartitionedCall¢&conv_block3d_1/StatefulPartitionedCall¢&conv_block3d_2/StatefulPartitionedCall¢&conv_block3d_3/StatefulPartitionedCall¢&conv_block3d_4/StatefulPartitionedCall¢&conv_block3d_5/StatefulPartitionedCallĘ
$conv_block3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv_block3d_1042419conv_block3d_1042421conv_block3d_1042423conv_block3d_1042425*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1041066ü
max_pooling3d/PartitionedCallPartitionedCall-conv_block3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_1041771ń
&conv_block3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv_block3d_2_1042429conv_block3d_2_1042431conv_block3d_2_1042433conv_block3d_2_1042435*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1041226
max_pooling3d_1/PartitionedCallPartitionedCall/conv_block3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:’’’’’’’’’@ @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_1041783ó
&conv_block3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv_block3d_4_1042439conv_block3d_4_1042441conv_block3d_4_1042443conv_block3d_4_1042445*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’@ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1041386
up_sampling3d_1/PartitionedCallPartitionedCall/conv_block3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’
@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_1042031Ŗ
concatenate_1/PartitionedCallPartitionedCall(up_sampling3d_1/PartitionedCall:output:0/conv_block3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’
@Ą* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1042040ń
&conv_block3d_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv_block3d_3_1042450conv_block3d_3_1042452conv_block3d_3_1042454conv_block3d_3_1042456*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1041706’
up_sampling3d/PartitionedCallPartitionedCall/conv_block3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_1042266¢
concatenate/PartitionedCallPartitionedCall&up_sampling3d/PartitionedCall:output:0-conv_block3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1042275š
&conv_block3d_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv_block3d_1_1042461conv_block3d_1_1042463conv_block3d_1_1042465conv_block3d_1_1042467*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1041546Ē
&conv_block3d_5/StatefulPartitionedCallStatefulPartitionedCall/conv_block3d_1/StatefulPartitionedCall:output:0conv_block3d_5_1042470conv_block3d_5_1042472*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1041847
IdentityIdentity/conv_block3d_5/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ŗ
NoOpNoOp%^conv_block3d/StatefulPartitionedCall'^conv_block3d_1/StatefulPartitionedCall'^conv_block3d_2/StatefulPartitionedCall'^conv_block3d_3/StatefulPartitionedCall'^conv_block3d_4/StatefulPartitionedCall'^conv_block3d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : 2L
$conv_block3d/StatefulPartitionedCall$conv_block3d/StatefulPartitionedCall2P
&conv_block3d_1/StatefulPartitionedCall&conv_block3d_1/StatefulPartitionedCall2P
&conv_block3d_2/StatefulPartitionedCall&conv_block3d_2/StatefulPartitionedCall2P
&conv_block3d_3/StatefulPartitionedCall&conv_block3d_3/StatefulPartitionedCall2P
&conv_block3d_4/StatefulPartitionedCall&conv_block3d_4/StatefulPartitionedCall2P
&conv_block3d_5/StatefulPartitionedCall&conv_block3d_5/StatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’
 
_user_specified_nameinputs
ó
M
1__inference_max_pooling3d_1_layer_call_fn_1044024

inputs
identityģ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_1041783
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’: {
W
_output_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ü
ģ
0__inference_conv_block3d_1_layer_call_fn_1041570
input_1%
unknown:` 
	unknown_0: '
	unknown_1:  
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1041546}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’`
!
_user_specified_name	input_1
Ģ


E__inference_conv3d_5_layer_call_and_return_conditional_losses_1044489

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’
@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:\ X
4
_output_shapes"
 :’’’’’’’’’
@@
 
_user_specified_nameinputs
Ī


E__inference_conv3d_8_layer_call_and_return_conditional_losses_1041299

inputs=
conv3d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ *
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’@ @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:’’’’’’’’’@ @
 
_user_specified_nameinputs
ų
ź
.__inference_conv_block3d_layer_call_fn_1041015
input_1%
unknown: 
	unknown_0: '
	unknown_1:  
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1041004}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’
!
_user_specified_name	input_1
Ł
h
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_1044029

inputs
identity½
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
ksize	
*
paddingSAME*
strides	

IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’: {
W
_output_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ź
Č
)__inference_u_net3d_layer_call_fn_1042572
input_1%
unknown: 
	unknown_0: '
	unknown_1:  
	unknown_2: '
	unknown_3: @
	unknown_4:@'
	unknown_5:@@
	unknown_6:@(
	unknown_7:@
	unknown_8:	)
	unknown_9:

unknown_10:	)

unknown_11:Ą@

unknown_12:@(

unknown_13:@@

unknown_14:@(

unknown_15:` 

unknown_16: (

unknown_17:  

unknown_18: (

unknown_19: 

unknown_20:
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_u_net3d_layer_call_and_return_conditional_losses_1042476}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’
!
_user_specified_name	input_1
ų
š
0__inference_conv_block3d_4_layer_call_fn_1043836

inputs&
unknown:@
	unknown_0:	)
	unknown_1:
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’@ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1041324|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’@ @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:’’’’’’’’’@ @
 
_user_specified_nameinputs

£
*__inference_conv3d_4_layer_call_fn_1044460

inputs%
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallģ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_4_layer_call_and_return_conditional_losses_1041139|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’
@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :’’’’’’’’’
@ 
 
_user_specified_nameinputs
Ņ


E__inference_conv3d_9_layer_call_and_return_conditional_losses_1044527

inputs>
conv3d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ *
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:\ X
4
_output_shapes"
 :’’’’’’’’’@ 
 
_user_specified_nameinputs
”
°
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1043699

inputsF
(conv3d_10_conv3d_readvariableop_resource: 7
)conv3d_10_biasadd_readvariableop_resource:
identity¢ conv3d_10/BiasAdd/ReadVariableOp¢conv3d_10/Conv3D/ReadVariableOp
conv3d_10/Conv3D/ReadVariableOpReadVariableOp(conv3d_10_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0“
conv3d_10/Conv3DConv3Dinputs'conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’*
paddingSAME*
strides	

 conv3d_10/BiasAdd/ReadVariableOpReadVariableOp)conv3d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0”
conv3d_10/BiasAddBiasAddconv3d_10/Conv3D:output:0(conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’w
IdentityIdentityconv3d_10/BiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’
NoOpNoOp!^conv3d_10/BiasAdd/ReadVariableOp ^conv3d_10/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 2D
 conv3d_10/BiasAdd/ReadVariableOp conv3d_10/BiasAdd/ReadVariableOp2B
conv3d_10/Conv3D/ReadVariableOpconv3d_10/Conv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’ 
 
_user_specified_nameinputs
Ļ


C__inference_conv3d_layer_call_and_return_conditional_losses_1040979

inputs<
conv3d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype0 
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’
 
_user_specified_nameinputs
ĒC
h
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_1044368

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0inputs*
T0*¶
_output_shapes£
 :’’’’’’’’’@ :’’’’’’’’’@ :’’’’’’’’’@ :’’’’’’’’’@ :’’’’’’’’’@ *
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4concat/axis:output:0*
N
*
T0*4
_output_shapes"
 :’’’’’’’’’
@ S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :š
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*
_output_shapes
:’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 *
	num_split@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :ā
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15split_1:output:16split_1:output:16split_1:output:17split_1:output:17split_1:output:18split_1:output:18split_1:output:19split_1:output:19split_1:output:20split_1:output:20split_1:output:21split_1:output:21split_1:output:22split_1:output:22split_1:output:23split_1:output:23split_1:output:24split_1:output:24split_1:output:25split_1:output:25split_1:output:26split_1:output:26split_1:output:27split_1:output:27split_1:output:28split_1:output:28split_1:output:29split_1:output:29split_1:output:30split_1:output:30split_1:output:31split_1:output:31split_1:output:32split_1:output:32split_1:output:33split_1:output:33split_1:output:34split_1:output:34split_1:output:35split_1:output:35split_1:output:36split_1:output:36split_1:output:37split_1:output:37split_1:output:38split_1:output:38split_1:output:39split_1:output:39split_1:output:40split_1:output:40split_1:output:41split_1:output:41split_1:output:42split_1:output:42split_1:output:43split_1:output:43split_1:output:44split_1:output:44split_1:output:45split_1:output:45split_1:output:46split_1:output:46split_1:output:47split_1:output:47split_1:output:48split_1:output:48split_1:output:49split_1:output:49split_1:output:50split_1:output:50split_1:output:51split_1:output:51split_1:output:52split_1:output:52split_1:output:53split_1:output:53split_1:output:54split_1:output:54split_1:output:55split_1:output:55split_1:output:56split_1:output:56split_1:output:57split_1:output:57split_1:output:58split_1:output:58split_1:output:59split_1:output:59split_1:output:60split_1:output:60split_1:output:61split_1:output:61split_1:output:62split_1:output:62split_1:output:63split_1:output:63concat_1/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’
 S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :	
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*¶
_output_shapes£
 :’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
*
	num_split O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :”

concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15split_2:output:16split_2:output:16split_2:output:17split_2:output:17split_2:output:18split_2:output:18split_2:output:19split_2:output:19split_2:output:20split_2:output:20split_2:output:21split_2:output:21split_2:output:22split_2:output:22split_2:output:23split_2:output:23split_2:output:24split_2:output:24split_2:output:25split_2:output:25split_2:output:26split_2:output:26split_2:output:27split_2:output:27split_2:output:28split_2:output:28split_2:output:29split_2:output:29split_2:output:30split_2:output:30split_2:output:31split_2:output:31concat_2/axis:output:0*
N@*
T0*5
_output_shapes#
!:’’’’’’’’’
@g
IdentityIdentityconcat_2:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’
@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :’’’’’’’’’@ :\ X
4
_output_shapes"
 :’’’’’’’’’@ 
 
_user_specified_nameinputs

t
H__inference_concatenate_layer_call_and_return_conditional_losses_1044381
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’`e
IdentityIdentityconcat:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:’’’’’’’’’@:’’’’’’’’’ :_ [
5
_output_shapes#
!:’’’’’’’’’@
"
_user_specified_name
inputs/0:_[
5
_output_shapes#
!:’’’’’’’’’ 
"
_user_specified_name
inputs/1
Ų
Ņ
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1041746
input_1/
conv3d_6_1041733:Ą@
conv3d_6_1041735:@.
conv3d_7_1041739:@@
conv3d_7_1041741:@
identity¢ conv3d_6/StatefulPartitionedCall¢ conv3d_7/StatefulPartitionedCall
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_6_1041733conv3d_6_1041735*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_6_layer_call_and_return_conditional_losses_1041619v
ReluRelu)conv3d_6/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_7_1041739conv3d_7_1041741*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_7_layer_call_and_return_conditional_losses_1041636x
Relu_1Relu)conv3d_7/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@p
IdentityIdentityRelu_1:activations:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
NoOpNoOp!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’
@Ą: : : : 2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’
@Ą
!
_user_specified_name	input_1
Ī=
Ź

D__inference_u_net3d_layer_call_and_return_conditional_losses_1042692
input_12
conv_block3d_1042635: "
conv_block3d_1042637: 2
conv_block3d_1042639:  "
conv_block3d_1042641: 4
conv_block3d_2_1042645: @$
conv_block3d_2_1042647:@4
conv_block3d_2_1042649:@@$
conv_block3d_2_1042651:@5
conv_block3d_4_1042655:@%
conv_block3d_4_1042657:	6
conv_block3d_4_1042659:%
conv_block3d_4_1042661:	5
conv_block3d_3_1042666:Ą@$
conv_block3d_3_1042668:@4
conv_block3d_3_1042670:@@$
conv_block3d_3_1042672:@4
conv_block3d_1_1042677:` $
conv_block3d_1_1042679: 4
conv_block3d_1_1042681:  $
conv_block3d_1_1042683: 4
conv_block3d_5_1042686: $
conv_block3d_5_1042688:
identity¢$conv_block3d/StatefulPartitionedCall¢&conv_block3d_1/StatefulPartitionedCall¢&conv_block3d_2/StatefulPartitionedCall¢&conv_block3d_3/StatefulPartitionedCall¢&conv_block3d_4/StatefulPartitionedCall¢&conv_block3d_5/StatefulPartitionedCallĒ
$conv_block3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv_block3d_1042635conv_block3d_1042637conv_block3d_1042639conv_block3d_1042641*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1041066ü
max_pooling3d/PartitionedCallPartitionedCall-conv_block3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_1041771ń
&conv_block3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv_block3d_2_1042645conv_block3d_2_1042647conv_block3d_2_1042649conv_block3d_2_1042651*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1041226
max_pooling3d_1/PartitionedCallPartitionedCall/conv_block3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:’’’’’’’’’@ @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_1041783ó
&conv_block3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv_block3d_4_1042655conv_block3d_4_1042657conv_block3d_4_1042659conv_block3d_4_1042661*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’@ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1041386
up_sampling3d_1/PartitionedCallPartitionedCall/conv_block3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’
@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_1042031Ŗ
concatenate_1/PartitionedCallPartitionedCall(up_sampling3d_1/PartitionedCall:output:0/conv_block3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’
@Ą* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1042040ń
&conv_block3d_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv_block3d_3_1042666conv_block3d_3_1042668conv_block3d_3_1042670conv_block3d_3_1042672*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1041706’
up_sampling3d/PartitionedCallPartitionedCall/conv_block3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_1042266¢
concatenate/PartitionedCallPartitionedCall&up_sampling3d/PartitionedCall:output:0-conv_block3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1042275š
&conv_block3d_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv_block3d_1_1042677conv_block3d_1_1042679conv_block3d_1_1042681conv_block3d_1_1042683*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1041546Ē
&conv_block3d_5/StatefulPartitionedCallStatefulPartitionedCall/conv_block3d_1/StatefulPartitionedCall:output:0conv_block3d_5_1042686conv_block3d_5_1042688*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1041847
IdentityIdentity/conv_block3d_5/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ŗ
NoOpNoOp%^conv_block3d/StatefulPartitionedCall'^conv_block3d_1/StatefulPartitionedCall'^conv_block3d_2/StatefulPartitionedCall'^conv_block3d_3/StatefulPartitionedCall'^conv_block3d_4/StatefulPartitionedCall'^conv_block3d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : 2L
$conv_block3d/StatefulPartitionedCall$conv_block3d/StatefulPartitionedCall2P
&conv_block3d_1/StatefulPartitionedCall&conv_block3d_1/StatefulPartitionedCall2P
&conv_block3d_2/StatefulPartitionedCall&conv_block3d_2/StatefulPartitionedCall2P
&conv_block3d_3/StatefulPartitionedCall&conv_block3d_3/StatefulPartitionedCall2P
&conv_block3d_4/StatefulPartitionedCall&conv_block3d_4/StatefulPartitionedCall2P
&conv_block3d_5/StatefulPartitionedCall&conv_block3d_5/StatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’
!
_user_specified_name	input_1
Ä
É
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1041106
input_1,
conv3d_1041093: 
conv3d_1041095: .
conv3d_1_1041099:  
conv3d_1_1041101: 
identity¢conv3d/StatefulPartitionedCall¢ conv3d_1/StatefulPartitionedCall’
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_1041093conv3d_1041095*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_conv3d_layer_call_and_return_conditional_losses_1040979u
ReluRelu'conv3d/StatefulPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_1_1041099conv3d_1_1041101*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_1_layer_call_and_return_conditional_losses_1040996y
Relu_1Relu)conv3d_1/StatefulPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ q
IdentityIdentityRelu_1:activations:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ 
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’: : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’
!
_user_specified_name	input_1
Ė=
É

D__inference_u_net3d_layer_call_and_return_conditional_losses_1042292

inputs2
conv_block3d_1041888: "
conv_block3d_1041890: 2
conv_block3d_1041892:  "
conv_block3d_1041894: 4
conv_block3d_2_1041898: @$
conv_block3d_2_1041900:@4
conv_block3d_2_1041902:@@$
conv_block3d_2_1041904:@5
conv_block3d_4_1041908:@%
conv_block3d_4_1041910:	6
conv_block3d_4_1041912:%
conv_block3d_4_1041914:	5
conv_block3d_3_1042042:Ą@$
conv_block3d_3_1042044:@4
conv_block3d_3_1042046:@@$
conv_block3d_3_1042048:@4
conv_block3d_1_1042277:` $
conv_block3d_1_1042279: 4
conv_block3d_1_1042281:  $
conv_block3d_1_1042283: 4
conv_block3d_5_1042286: $
conv_block3d_5_1042288:
identity¢$conv_block3d/StatefulPartitionedCall¢&conv_block3d_1/StatefulPartitionedCall¢&conv_block3d_2/StatefulPartitionedCall¢&conv_block3d_3/StatefulPartitionedCall¢&conv_block3d_4/StatefulPartitionedCall¢&conv_block3d_5/StatefulPartitionedCallĘ
$conv_block3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv_block3d_1041888conv_block3d_1041890conv_block3d_1041892conv_block3d_1041894*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1041004ü
max_pooling3d/PartitionedCallPartitionedCall-conv_block3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_1041771ń
&conv_block3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv_block3d_2_1041898conv_block3d_2_1041900conv_block3d_2_1041902conv_block3d_2_1041904*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1041164
max_pooling3d_1/PartitionedCallPartitionedCall/conv_block3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:’’’’’’’’’@ @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_1041783ó
&conv_block3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv_block3d_4_1041908conv_block3d_4_1041910conv_block3d_4_1041912conv_block3d_4_1041914*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’@ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1041324
up_sampling3d_1/PartitionedCallPartitionedCall/conv_block3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’
@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_1042031Ŗ
concatenate_1/PartitionedCallPartitionedCall(up_sampling3d_1/PartitionedCall:output:0/conv_block3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’
@Ą* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1042040ń
&conv_block3d_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv_block3d_3_1042042conv_block3d_3_1042044conv_block3d_3_1042046conv_block3d_3_1042048*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1041644’
up_sampling3d/PartitionedCallPartitionedCall/conv_block3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_1042266¢
concatenate/PartitionedCallPartitionedCall&up_sampling3d/PartitionedCall:output:0-conv_block3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1042275š
&conv_block3d_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv_block3d_1_1042277conv_block3d_1_1042279conv_block3d_1_1042281conv_block3d_1_1042283*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1041484Ē
&conv_block3d_5/StatefulPartitionedCallStatefulPartitionedCall/conv_block3d_1/StatefulPartitionedCall:output:0conv_block3d_5_1042286conv_block3d_5_1042288*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1041810
IdentityIdentity/conv_block3d_5/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ŗ
NoOpNoOp%^conv_block3d/StatefulPartitionedCall'^conv_block3d_1/StatefulPartitionedCall'^conv_block3d_2/StatefulPartitionedCall'^conv_block3d_3/StatefulPartitionedCall'^conv_block3d_4/StatefulPartitionedCall'^conv_block3d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : 2L
$conv_block3d/StatefulPartitionedCall$conv_block3d/StatefulPartitionedCall2P
&conv_block3d_1/StatefulPartitionedCall&conv_block3d_1/StatefulPartitionedCall2P
&conv_block3d_2/StatefulPartitionedCall&conv_block3d_2/StatefulPartitionedCall2P
&conv_block3d_3/StatefulPartitionedCall&conv_block3d_3/StatefulPartitionedCall2P
&conv_block3d_4/StatefulPartitionedCall&conv_block3d_4/StatefulPartitionedCall2P
&conv_block3d_5/StatefulPartitionedCall&conv_block3d_5/StatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’
 
_user_specified_nameinputs
ų
ģ
0__inference_conv_block3d_3_layer_call_fn_1043960

inputs&
unknown:Ą@
	unknown_0:@'
	unknown_1:@@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1041644|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’
@Ą: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’
@Ą
 
_user_specified_nameinputs
Žy
f
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_1042266

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :£
splitSplitsplit/split_dim:output:0inputs*
T0*Ö
_output_shapesĆ
Ą:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@*
	num_split
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :°
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9concat/axis:output:0*
N*
T0*4
_output_shapes"
 :’’’’’’’’’@@S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ń
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*
_output_shapes
:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :'
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15split_1:output:16split_1:output:16split_1:output:17split_1:output:17split_1:output:18split_1:output:18split_1:output:19split_1:output:19split_1:output:20split_1:output:20split_1:output:21split_1:output:21split_1:output:22split_1:output:22split_1:output:23split_1:output:23split_1:output:24split_1:output:24split_1:output:25split_1:output:25split_1:output:26split_1:output:26split_1:output:27split_1:output:27split_1:output:28split_1:output:28split_1:output:29split_1:output:29split_1:output:30split_1:output:30split_1:output:31split_1:output:31split_1:output:32split_1:output:32split_1:output:33split_1:output:33split_1:output:34split_1:output:34split_1:output:35split_1:output:35split_1:output:36split_1:output:36split_1:output:37split_1:output:37split_1:output:38split_1:output:38split_1:output:39split_1:output:39split_1:output:40split_1:output:40split_1:output:41split_1:output:41split_1:output:42split_1:output:42split_1:output:43split_1:output:43split_1:output:44split_1:output:44split_1:output:45split_1:output:45split_1:output:46split_1:output:46split_1:output:47split_1:output:47split_1:output:48split_1:output:48split_1:output:49split_1:output:49split_1:output:50split_1:output:50split_1:output:51split_1:output:51split_1:output:52split_1:output:52split_1:output:53split_1:output:53split_1:output:54split_1:output:54split_1:output:55split_1:output:55split_1:output:56split_1:output:56split_1:output:57split_1:output:57split_1:output:58split_1:output:58split_1:output:59split_1:output:59split_1:output:60split_1:output:60split_1:output:61split_1:output:61split_1:output:62split_1:output:62split_1:output:63split_1:output:63split_1:output:64split_1:output:64split_1:output:65split_1:output:65split_1:output:66split_1:output:66split_1:output:67split_1:output:67split_1:output:68split_1:output:68split_1:output:69split_1:output:69split_1:output:70split_1:output:70split_1:output:71split_1:output:71split_1:output:72split_1:output:72split_1:output:73split_1:output:73split_1:output:74split_1:output:74split_1:output:75split_1:output:75split_1:output:76split_1:output:76split_1:output:77split_1:output:77split_1:output:78split_1:output:78split_1:output:79split_1:output:79split_1:output:80split_1:output:80split_1:output:81split_1:output:81split_1:output:82split_1:output:82split_1:output:83split_1:output:83split_1:output:84split_1:output:84split_1:output:85split_1:output:85split_1:output:86split_1:output:86split_1:output:87split_1:output:87split_1:output:88split_1:output:88split_1:output:89split_1:output:89split_1:output:90split_1:output:90split_1:output:91split_1:output:91split_1:output:92split_1:output:92split_1:output:93split_1:output:93split_1:output:94split_1:output:94split_1:output:95split_1:output:95split_1:output:96split_1:output:96split_1:output:97split_1:output:97split_1:output:98split_1:output:98split_1:output:99split_1:output:99split_1:output:100split_1:output:100split_1:output:101split_1:output:101split_1:output:102split_1:output:102split_1:output:103split_1:output:103split_1:output:104split_1:output:104split_1:output:105split_1:output:105split_1:output:106split_1:output:106split_1:output:107split_1:output:107split_1:output:108split_1:output:108split_1:output:109split_1:output:109split_1:output:110split_1:output:110split_1:output:111split_1:output:111split_1:output:112split_1:output:112split_1:output:113split_1:output:113split_1:output:114split_1:output:114split_1:output:115split_1:output:115split_1:output:116split_1:output:116split_1:output:117split_1:output:117split_1:output:118split_1:output:118split_1:output:119split_1:output:119split_1:output:120split_1:output:120split_1:output:121split_1:output:121split_1:output:122split_1:output:122split_1:output:123split_1:output:123split_1:output:124split_1:output:124split_1:output:125split_1:output:125split_1:output:126split_1:output:126split_1:output:127split_1:output:127concat_1/axis:output:0*
N*
T0*4
_output_shapes"
 :’’’’’’’’’@@S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ņ
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*
_output_shapes
:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@*
	num_split@O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :ā
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15split_2:output:16split_2:output:16split_2:output:17split_2:output:17split_2:output:18split_2:output:18split_2:output:19split_2:output:19split_2:output:20split_2:output:20split_2:output:21split_2:output:21split_2:output:22split_2:output:22split_2:output:23split_2:output:23split_2:output:24split_2:output:24split_2:output:25split_2:output:25split_2:output:26split_2:output:26split_2:output:27split_2:output:27split_2:output:28split_2:output:28split_2:output:29split_2:output:29split_2:output:30split_2:output:30split_2:output:31split_2:output:31split_2:output:32split_2:output:32split_2:output:33split_2:output:33split_2:output:34split_2:output:34split_2:output:35split_2:output:35split_2:output:36split_2:output:36split_2:output:37split_2:output:37split_2:output:38split_2:output:38split_2:output:39split_2:output:39split_2:output:40split_2:output:40split_2:output:41split_2:output:41split_2:output:42split_2:output:42split_2:output:43split_2:output:43split_2:output:44split_2:output:44split_2:output:45split_2:output:45split_2:output:46split_2:output:46split_2:output:47split_2:output:47split_2:output:48split_2:output:48split_2:output:49split_2:output:49split_2:output:50split_2:output:50split_2:output:51split_2:output:51split_2:output:52split_2:output:52split_2:output:53split_2:output:53split_2:output:54split_2:output:54split_2:output:55split_2:output:55split_2:output:56split_2:output:56split_2:output:57split_2:output:57split_2:output:58split_2:output:58split_2:output:59split_2:output:59split_2:output:60split_2:output:60split_2:output:61split_2:output:61split_2:output:62split_2:output:62split_2:output:63split_2:output:63concat_2/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’@g
IdentityIdentityconcat_2:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :’’’’’’’’’
@@:\ X
4
_output_shapes"
 :’’’’’’’’’
@@
 
_user_specified_nameinputs
ü
ģ
0__inference_conv_block3d_1_layer_call_fn_1041495
input_1%
unknown:` 
	unknown_0: '
	unknown_1:  
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1041484}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’`
!
_user_specified_name	input_1
Õ
Ń
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1041266
input_1.
conv3d_4_1041253: @
conv3d_4_1041255:@.
conv3d_5_1041259:@@
conv3d_5_1041261:@
identity¢ conv3d_4/StatefulPartitionedCall¢ conv3d_5/StatefulPartitionedCall
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_4_1041253conv3d_4_1041255*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_4_layer_call_and_return_conditional_losses_1041139v
ReluRelu)conv3d_4/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_5_1041259conv3d_5_1041261*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_5_layer_call_and_return_conditional_losses_1041156x
Relu_1Relu)conv3d_5/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@p
IdentityIdentityRelu_1:activations:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
NoOpNoOp!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:’’’’’’’’’
@ : : : : 2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall:] Y
4
_output_shapes"
 :’’’’’’’’’
@ 
!
_user_specified_name	input_1
	
į
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1041881
input_1/
conv3d_10_1041875: 
conv3d_10_1041877:
identity¢!conv3d_10/StatefulPartitionedCall
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_10_1041875conv3d_10_1041877*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv3d_10_layer_call_and_return_conditional_losses_1041803
IdentityIdentity*conv3d_10/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’j
NoOpNoOp"^conv3d_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’ 
!
_user_specified_name	input_1

£
*__inference_conv3d_5_layer_call_fn_1044479

inputs%
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallģ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_5_layer_call_and_return_conditional_losses_1041156|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’
@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :’’’’’’’’’
@@
 
_user_specified_nameinputs
ū
ķ
0__inference_conv_block3d_3_layer_call_fn_1041655
input_1&
unknown:Ą@
	unknown_0:@'
	unknown_1:@@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1041644|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’
@Ą: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’
@Ą
!
_user_specified_name	input_1

[
/__inference_concatenate_1_layer_call_fn_1044387
inputs_0
inputs_1
identityÕ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’
@Ą* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1042040n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’
@Ą"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:’’’’’’’’’
@:’’’’’’’’’
@@:_ [
5
_output_shapes#
!:’’’’’’’’’
@
"
_user_specified_name
inputs/0:^Z
4
_output_shapes"
 :’’’’’’’’’
@@
"
_user_specified_name
inputs/1
ä
K
/__inference_up_sampling3d_layer_call_fn_1044034

inputs
identityČ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_1042266n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :’’’’’’’’’
@@:\ X
4
_output_shapes"
 :’’’’’’’’’
@@
 
_user_specified_nameinputs
ļ
K
/__inference_max_pooling3d_layer_call_fn_1044014

inputs
identityź
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_1041771
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’: {
W
_output_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Š
æ
D__inference_u_net3d_layer_call_and_return_conditional_losses_1043254

inputsP
2conv_block3d_conv3d_conv3d_readvariableop_resource: A
3conv_block3d_conv3d_biasadd_readvariableop_resource: R
4conv_block3d_conv3d_1_conv3d_readvariableop_resource:  C
5conv_block3d_conv3d_1_biasadd_readvariableop_resource: T
6conv_block3d_2_conv3d_4_conv3d_readvariableop_resource: @E
7conv_block3d_2_conv3d_4_biasadd_readvariableop_resource:@T
6conv_block3d_2_conv3d_5_conv3d_readvariableop_resource:@@E
7conv_block3d_2_conv3d_5_biasadd_readvariableop_resource:@U
6conv_block3d_4_conv3d_8_conv3d_readvariableop_resource:@F
7conv_block3d_4_conv3d_8_biasadd_readvariableop_resource:	V
6conv_block3d_4_conv3d_9_conv3d_readvariableop_resource:F
7conv_block3d_4_conv3d_9_biasadd_readvariableop_resource:	U
6conv_block3d_3_conv3d_6_conv3d_readvariableop_resource:Ą@E
7conv_block3d_3_conv3d_6_biasadd_readvariableop_resource:@T
6conv_block3d_3_conv3d_7_conv3d_readvariableop_resource:@@E
7conv_block3d_3_conv3d_7_biasadd_readvariableop_resource:@T
6conv_block3d_1_conv3d_2_conv3d_readvariableop_resource:` E
7conv_block3d_1_conv3d_2_biasadd_readvariableop_resource: T
6conv_block3d_1_conv3d_3_conv3d_readvariableop_resource:  E
7conv_block3d_1_conv3d_3_biasadd_readvariableop_resource: U
7conv_block3d_5_conv3d_10_conv3d_readvariableop_resource: F
8conv_block3d_5_conv3d_10_biasadd_readvariableop_resource:
identity¢*conv_block3d/conv3d/BiasAdd/ReadVariableOp¢)conv_block3d/conv3d/Conv3D/ReadVariableOp¢,conv_block3d/conv3d_1/BiasAdd/ReadVariableOp¢+conv_block3d/conv3d_1/Conv3D/ReadVariableOp¢.conv_block3d_1/conv3d_2/BiasAdd/ReadVariableOp¢-conv_block3d_1/conv3d_2/Conv3D/ReadVariableOp¢.conv_block3d_1/conv3d_3/BiasAdd/ReadVariableOp¢-conv_block3d_1/conv3d_3/Conv3D/ReadVariableOp¢.conv_block3d_2/conv3d_4/BiasAdd/ReadVariableOp¢-conv_block3d_2/conv3d_4/Conv3D/ReadVariableOp¢.conv_block3d_2/conv3d_5/BiasAdd/ReadVariableOp¢-conv_block3d_2/conv3d_5/Conv3D/ReadVariableOp¢.conv_block3d_3/conv3d_6/BiasAdd/ReadVariableOp¢-conv_block3d_3/conv3d_6/Conv3D/ReadVariableOp¢.conv_block3d_3/conv3d_7/BiasAdd/ReadVariableOp¢-conv_block3d_3/conv3d_7/Conv3D/ReadVariableOp¢.conv_block3d_4/conv3d_8/BiasAdd/ReadVariableOp¢-conv_block3d_4/conv3d_8/Conv3D/ReadVariableOp¢.conv_block3d_4/conv3d_9/BiasAdd/ReadVariableOp¢-conv_block3d_4/conv3d_9/Conv3D/ReadVariableOp¢/conv_block3d_5/conv3d_10/BiasAdd/ReadVariableOp¢.conv_block3d_5/conv3d_10/Conv3D/ReadVariableOpØ
)conv_block3d/conv3d/Conv3D/ReadVariableOpReadVariableOp2conv_block3d_conv3d_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0Č
conv_block3d/conv3d/Conv3DConv3Dinputs1conv_block3d/conv3d/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	

*conv_block3d/conv3d/BiasAdd/ReadVariableOpReadVariableOp3conv_block3d_conv3d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0æ
conv_block3d/conv3d/BiasAddBiasAdd#conv_block3d/conv3d/Conv3D:output:02conv_block3d/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
conv_block3d/ReluRelu$conv_block3d/conv3d/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ ¬
+conv_block3d/conv3d_1/Conv3D/ReadVariableOpReadVariableOp4conv_block3d_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0å
conv_block3d/conv3d_1/Conv3DConv3Dconv_block3d/Relu:activations:03conv_block3d/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	

,conv_block3d/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp5conv_block3d_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Å
conv_block3d/conv3d_1/BiasAddBiasAdd%conv_block3d/conv3d_1/Conv3D:output:04conv_block3d/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
conv_block3d/Relu_1Relu&conv_block3d/conv3d_1/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ Ć
max_pooling3d/MaxPool3D	MaxPool3D!conv_block3d/Relu_1:activations:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@ *
ksize	
*
paddingSAME*
strides	
°
-conv_block3d_2/conv3d_4/Conv3D/ReadVariableOpReadVariableOp6conv_block3d_2_conv3d_4_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0é
conv_block3d_2/conv3d_4/Conv3DConv3D max_pooling3d/MaxPool3D:output:05conv_block3d_2/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
¢
.conv_block3d_2/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp7conv_block3d_2_conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ź
conv_block3d_2/conv3d_4/BiasAddBiasAdd'conv_block3d_2/conv3d_4/Conv3D:output:06conv_block3d_2/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
conv_block3d_2/ReluRelu(conv_block3d_2/conv3d_4/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@°
-conv_block3d_2/conv3d_5/Conv3D/ReadVariableOpReadVariableOp6conv_block3d_2_conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0ź
conv_block3d_2/conv3d_5/Conv3DConv3D!conv_block3d_2/Relu:activations:05conv_block3d_2/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
¢
.conv_block3d_2/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp7conv_block3d_2_conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ź
conv_block3d_2/conv3d_5/BiasAddBiasAdd'conv_block3d_2/conv3d_5/Conv3D:output:06conv_block3d_2/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
conv_block3d_2/Relu_1Relu(conv_block3d_2/conv3d_5/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@Ę
max_pooling3d_1/MaxPool3D	MaxPool3D#conv_block3d_2/Relu_1:activations:0*
T0*3
_output_shapes!
:’’’’’’’’’@ @*
ksize	
*
paddingSAME*
strides	
±
-conv_block3d_4/conv3d_8/Conv3D/ReadVariableOpReadVariableOp6conv_block3d_4_conv3d_8_conv3d_readvariableop_resource*+
_output_shapes
:@*
dtype0ė
conv_block3d_4/conv3d_8/Conv3DConv3D"max_pooling3d_1/MaxPool3D:output:05conv_block3d_4/conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ *
paddingSAME*
strides	
£
.conv_block3d_4/conv3d_8/BiasAdd/ReadVariableOpReadVariableOp7conv_block3d_4_conv3d_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ź
conv_block3d_4/conv3d_8/BiasAddBiasAdd'conv_block3d_4/conv3d_8/Conv3D:output:06conv_block3d_4/conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ 
conv_block3d_4/ReluRelu(conv_block3d_4/conv3d_8/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ ²
-conv_block3d_4/conv3d_9/Conv3D/ReadVariableOpReadVariableOp6conv_block3d_4_conv3d_9_conv3d_readvariableop_resource*,
_output_shapes
:*
dtype0ź
conv_block3d_4/conv3d_9/Conv3DConv3D!conv_block3d_4/Relu:activations:05conv_block3d_4/conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ *
paddingSAME*
strides	
£
.conv_block3d_4/conv3d_9/BiasAdd/ReadVariableOpReadVariableOp7conv_block3d_4_conv3d_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ź
conv_block3d_4/conv3d_9/BiasAddBiasAdd'conv_block3d_4/conv3d_9/Conv3D:output:06conv_block3d_4/conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ 
conv_block3d_4/Relu_1Relu(conv_block3d_4/conv3d_9/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ a
up_sampling3d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ą
up_sampling3d_1/splitSplit(up_sampling3d_1/split/split_dim:output:0#conv_block3d_4/Relu_1:activations:0*
T0*¶
_output_shapes£
 :’’’’’’’’’@ :’’’’’’’’’@ :’’’’’’’’’@ :’’’’’’’’’@ :’’’’’’’’’@ *
	num_split]
up_sampling3d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Š
up_sampling3d_1/concatConcatV2up_sampling3d_1/split:output:0up_sampling3d_1/split:output:0up_sampling3d_1/split:output:1up_sampling3d_1/split:output:1up_sampling3d_1/split:output:2up_sampling3d_1/split:output:2up_sampling3d_1/split:output:3up_sampling3d_1/split:output:3up_sampling3d_1/split:output:4up_sampling3d_1/split:output:4$up_sampling3d_1/concat/axis:output:0*
N
*
T0*4
_output_shapes"
 :’’’’’’’’’
@ c
!up_sampling3d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
up_sampling3d_1/split_1Split*up_sampling3d_1/split_1/split_dim:output:0up_sampling3d_1/concat:output:0*
T0*
_output_shapes
:’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 *
	num_split@_
up_sampling3d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :$
up_sampling3d_1/concat_1ConcatV2 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:5 up_sampling3d_1/split_1:output:5 up_sampling3d_1/split_1:output:6 up_sampling3d_1/split_1:output:6 up_sampling3d_1/split_1:output:7 up_sampling3d_1/split_1:output:7 up_sampling3d_1/split_1:output:8 up_sampling3d_1/split_1:output:8 up_sampling3d_1/split_1:output:9 up_sampling3d_1/split_1:output:9!up_sampling3d_1/split_1:output:10!up_sampling3d_1/split_1:output:10!up_sampling3d_1/split_1:output:11!up_sampling3d_1/split_1:output:11!up_sampling3d_1/split_1:output:12!up_sampling3d_1/split_1:output:12!up_sampling3d_1/split_1:output:13!up_sampling3d_1/split_1:output:13!up_sampling3d_1/split_1:output:14!up_sampling3d_1/split_1:output:14!up_sampling3d_1/split_1:output:15!up_sampling3d_1/split_1:output:15!up_sampling3d_1/split_1:output:16!up_sampling3d_1/split_1:output:16!up_sampling3d_1/split_1:output:17!up_sampling3d_1/split_1:output:17!up_sampling3d_1/split_1:output:18!up_sampling3d_1/split_1:output:18!up_sampling3d_1/split_1:output:19!up_sampling3d_1/split_1:output:19!up_sampling3d_1/split_1:output:20!up_sampling3d_1/split_1:output:20!up_sampling3d_1/split_1:output:21!up_sampling3d_1/split_1:output:21!up_sampling3d_1/split_1:output:22!up_sampling3d_1/split_1:output:22!up_sampling3d_1/split_1:output:23!up_sampling3d_1/split_1:output:23!up_sampling3d_1/split_1:output:24!up_sampling3d_1/split_1:output:24!up_sampling3d_1/split_1:output:25!up_sampling3d_1/split_1:output:25!up_sampling3d_1/split_1:output:26!up_sampling3d_1/split_1:output:26!up_sampling3d_1/split_1:output:27!up_sampling3d_1/split_1:output:27!up_sampling3d_1/split_1:output:28!up_sampling3d_1/split_1:output:28!up_sampling3d_1/split_1:output:29!up_sampling3d_1/split_1:output:29!up_sampling3d_1/split_1:output:30!up_sampling3d_1/split_1:output:30!up_sampling3d_1/split_1:output:31!up_sampling3d_1/split_1:output:31!up_sampling3d_1/split_1:output:32!up_sampling3d_1/split_1:output:32!up_sampling3d_1/split_1:output:33!up_sampling3d_1/split_1:output:33!up_sampling3d_1/split_1:output:34!up_sampling3d_1/split_1:output:34!up_sampling3d_1/split_1:output:35!up_sampling3d_1/split_1:output:35!up_sampling3d_1/split_1:output:36!up_sampling3d_1/split_1:output:36!up_sampling3d_1/split_1:output:37!up_sampling3d_1/split_1:output:37!up_sampling3d_1/split_1:output:38!up_sampling3d_1/split_1:output:38!up_sampling3d_1/split_1:output:39!up_sampling3d_1/split_1:output:39!up_sampling3d_1/split_1:output:40!up_sampling3d_1/split_1:output:40!up_sampling3d_1/split_1:output:41!up_sampling3d_1/split_1:output:41!up_sampling3d_1/split_1:output:42!up_sampling3d_1/split_1:output:42!up_sampling3d_1/split_1:output:43!up_sampling3d_1/split_1:output:43!up_sampling3d_1/split_1:output:44!up_sampling3d_1/split_1:output:44!up_sampling3d_1/split_1:output:45!up_sampling3d_1/split_1:output:45!up_sampling3d_1/split_1:output:46!up_sampling3d_1/split_1:output:46!up_sampling3d_1/split_1:output:47!up_sampling3d_1/split_1:output:47!up_sampling3d_1/split_1:output:48!up_sampling3d_1/split_1:output:48!up_sampling3d_1/split_1:output:49!up_sampling3d_1/split_1:output:49!up_sampling3d_1/split_1:output:50!up_sampling3d_1/split_1:output:50!up_sampling3d_1/split_1:output:51!up_sampling3d_1/split_1:output:51!up_sampling3d_1/split_1:output:52!up_sampling3d_1/split_1:output:52!up_sampling3d_1/split_1:output:53!up_sampling3d_1/split_1:output:53!up_sampling3d_1/split_1:output:54!up_sampling3d_1/split_1:output:54!up_sampling3d_1/split_1:output:55!up_sampling3d_1/split_1:output:55!up_sampling3d_1/split_1:output:56!up_sampling3d_1/split_1:output:56!up_sampling3d_1/split_1:output:57!up_sampling3d_1/split_1:output:57!up_sampling3d_1/split_1:output:58!up_sampling3d_1/split_1:output:58!up_sampling3d_1/split_1:output:59!up_sampling3d_1/split_1:output:59!up_sampling3d_1/split_1:output:60!up_sampling3d_1/split_1:output:60!up_sampling3d_1/split_1:output:61!up_sampling3d_1/split_1:output:61!up_sampling3d_1/split_1:output:62!up_sampling3d_1/split_1:output:62!up_sampling3d_1/split_1:output:63!up_sampling3d_1/split_1:output:63&up_sampling3d_1/concat_1/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’
 c
!up_sampling3d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ā	
up_sampling3d_1/split_2Split*up_sampling3d_1/split_2/split_dim:output:0!up_sampling3d_1/concat_1:output:0*
T0*¶
_output_shapes£
 :’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
*
	num_split _
up_sampling3d_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :Į
up_sampling3d_1/concat_2ConcatV2 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:5 up_sampling3d_1/split_2:output:5 up_sampling3d_1/split_2:output:6 up_sampling3d_1/split_2:output:6 up_sampling3d_1/split_2:output:7 up_sampling3d_1/split_2:output:7 up_sampling3d_1/split_2:output:8 up_sampling3d_1/split_2:output:8 up_sampling3d_1/split_2:output:9 up_sampling3d_1/split_2:output:9!up_sampling3d_1/split_2:output:10!up_sampling3d_1/split_2:output:10!up_sampling3d_1/split_2:output:11!up_sampling3d_1/split_2:output:11!up_sampling3d_1/split_2:output:12!up_sampling3d_1/split_2:output:12!up_sampling3d_1/split_2:output:13!up_sampling3d_1/split_2:output:13!up_sampling3d_1/split_2:output:14!up_sampling3d_1/split_2:output:14!up_sampling3d_1/split_2:output:15!up_sampling3d_1/split_2:output:15!up_sampling3d_1/split_2:output:16!up_sampling3d_1/split_2:output:16!up_sampling3d_1/split_2:output:17!up_sampling3d_1/split_2:output:17!up_sampling3d_1/split_2:output:18!up_sampling3d_1/split_2:output:18!up_sampling3d_1/split_2:output:19!up_sampling3d_1/split_2:output:19!up_sampling3d_1/split_2:output:20!up_sampling3d_1/split_2:output:20!up_sampling3d_1/split_2:output:21!up_sampling3d_1/split_2:output:21!up_sampling3d_1/split_2:output:22!up_sampling3d_1/split_2:output:22!up_sampling3d_1/split_2:output:23!up_sampling3d_1/split_2:output:23!up_sampling3d_1/split_2:output:24!up_sampling3d_1/split_2:output:24!up_sampling3d_1/split_2:output:25!up_sampling3d_1/split_2:output:25!up_sampling3d_1/split_2:output:26!up_sampling3d_1/split_2:output:26!up_sampling3d_1/split_2:output:27!up_sampling3d_1/split_2:output:27!up_sampling3d_1/split_2:output:28!up_sampling3d_1/split_2:output:28!up_sampling3d_1/split_2:output:29!up_sampling3d_1/split_2:output:29!up_sampling3d_1/split_2:output:30!up_sampling3d_1/split_2:output:30!up_sampling3d_1/split_2:output:31!up_sampling3d_1/split_2:output:31&up_sampling3d_1/concat_2/axis:output:0*
N@*
T0*5
_output_shapes#
!:’’’’’’’’’
@[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Õ
concatenate_1/concatConcatV2!up_sampling3d_1/concat_2:output:0#conv_block3d_2/Relu_1:activations:0"concatenate_1/concat/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’
@Ą±
-conv_block3d_3/conv3d_6/Conv3D/ReadVariableOpReadVariableOp6conv_block3d_3_conv3d_6_conv3d_readvariableop_resource*+
_output_shapes
:Ą@*
dtype0ę
conv_block3d_3/conv3d_6/Conv3DConv3Dconcatenate_1/concat:output:05conv_block3d_3/conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
¢
.conv_block3d_3/conv3d_6/BiasAdd/ReadVariableOpReadVariableOp7conv_block3d_3_conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ź
conv_block3d_3/conv3d_6/BiasAddBiasAdd'conv_block3d_3/conv3d_6/Conv3D:output:06conv_block3d_3/conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
conv_block3d_3/ReluRelu(conv_block3d_3/conv3d_6/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@°
-conv_block3d_3/conv3d_7/Conv3D/ReadVariableOpReadVariableOp6conv_block3d_3_conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0ź
conv_block3d_3/conv3d_7/Conv3DConv3D!conv_block3d_3/Relu:activations:05conv_block3d_3/conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
¢
.conv_block3d_3/conv3d_7/BiasAdd/ReadVariableOpReadVariableOp7conv_block3d_3_conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ź
conv_block3d_3/conv3d_7/BiasAddBiasAdd'conv_block3d_3/conv3d_7/Conv3D:output:06conv_block3d_3/conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
conv_block3d_3/Relu_1Relu(conv_block3d_3/conv3d_7/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@_
up_sampling3d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ü
up_sampling3d/splitSplit&up_sampling3d/split/split_dim:output:0#conv_block3d_3/Relu_1:activations:0*
T0*Ö
_output_shapesĆ
Ą:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@*
	num_split
[
up_sampling3d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ä
up_sampling3d/concatConcatV2up_sampling3d/split:output:0up_sampling3d/split:output:0up_sampling3d/split:output:1up_sampling3d/split:output:1up_sampling3d/split:output:2up_sampling3d/split:output:2up_sampling3d/split:output:3up_sampling3d/split:output:3up_sampling3d/split:output:4up_sampling3d/split:output:4up_sampling3d/split:output:5up_sampling3d/split:output:5up_sampling3d/split:output:6up_sampling3d/split:output:6up_sampling3d/split:output:7up_sampling3d/split:output:7up_sampling3d/split:output:8up_sampling3d/split:output:8up_sampling3d/split:output:9up_sampling3d/split:output:9"up_sampling3d/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :’’’’’’’’’@@a
up_sampling3d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
up_sampling3d/split_1Split(up_sampling3d/split_1/split_dim:output:0up_sampling3d/concat:output:0*
T0*
_output_shapes
:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@*
	num_split]
up_sampling3d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :µC
up_sampling3d/concat_1ConcatV2up_sampling3d/split_1:output:0up_sampling3d/split_1:output:0up_sampling3d/split_1:output:1up_sampling3d/split_1:output:1up_sampling3d/split_1:output:2up_sampling3d/split_1:output:2up_sampling3d/split_1:output:3up_sampling3d/split_1:output:3up_sampling3d/split_1:output:4up_sampling3d/split_1:output:4up_sampling3d/split_1:output:5up_sampling3d/split_1:output:5up_sampling3d/split_1:output:6up_sampling3d/split_1:output:6up_sampling3d/split_1:output:7up_sampling3d/split_1:output:7up_sampling3d/split_1:output:8up_sampling3d/split_1:output:8up_sampling3d/split_1:output:9up_sampling3d/split_1:output:9up_sampling3d/split_1:output:10up_sampling3d/split_1:output:10up_sampling3d/split_1:output:11up_sampling3d/split_1:output:11up_sampling3d/split_1:output:12up_sampling3d/split_1:output:12up_sampling3d/split_1:output:13up_sampling3d/split_1:output:13up_sampling3d/split_1:output:14up_sampling3d/split_1:output:14up_sampling3d/split_1:output:15up_sampling3d/split_1:output:15up_sampling3d/split_1:output:16up_sampling3d/split_1:output:16up_sampling3d/split_1:output:17up_sampling3d/split_1:output:17up_sampling3d/split_1:output:18up_sampling3d/split_1:output:18up_sampling3d/split_1:output:19up_sampling3d/split_1:output:19up_sampling3d/split_1:output:20up_sampling3d/split_1:output:20up_sampling3d/split_1:output:21up_sampling3d/split_1:output:21up_sampling3d/split_1:output:22up_sampling3d/split_1:output:22up_sampling3d/split_1:output:23up_sampling3d/split_1:output:23up_sampling3d/split_1:output:24up_sampling3d/split_1:output:24up_sampling3d/split_1:output:25up_sampling3d/split_1:output:25up_sampling3d/split_1:output:26up_sampling3d/split_1:output:26up_sampling3d/split_1:output:27up_sampling3d/split_1:output:27up_sampling3d/split_1:output:28up_sampling3d/split_1:output:28up_sampling3d/split_1:output:29up_sampling3d/split_1:output:29up_sampling3d/split_1:output:30up_sampling3d/split_1:output:30up_sampling3d/split_1:output:31up_sampling3d/split_1:output:31up_sampling3d/split_1:output:32up_sampling3d/split_1:output:32up_sampling3d/split_1:output:33up_sampling3d/split_1:output:33up_sampling3d/split_1:output:34up_sampling3d/split_1:output:34up_sampling3d/split_1:output:35up_sampling3d/split_1:output:35up_sampling3d/split_1:output:36up_sampling3d/split_1:output:36up_sampling3d/split_1:output:37up_sampling3d/split_1:output:37up_sampling3d/split_1:output:38up_sampling3d/split_1:output:38up_sampling3d/split_1:output:39up_sampling3d/split_1:output:39up_sampling3d/split_1:output:40up_sampling3d/split_1:output:40up_sampling3d/split_1:output:41up_sampling3d/split_1:output:41up_sampling3d/split_1:output:42up_sampling3d/split_1:output:42up_sampling3d/split_1:output:43up_sampling3d/split_1:output:43up_sampling3d/split_1:output:44up_sampling3d/split_1:output:44up_sampling3d/split_1:output:45up_sampling3d/split_1:output:45up_sampling3d/split_1:output:46up_sampling3d/split_1:output:46up_sampling3d/split_1:output:47up_sampling3d/split_1:output:47up_sampling3d/split_1:output:48up_sampling3d/split_1:output:48up_sampling3d/split_1:output:49up_sampling3d/split_1:output:49up_sampling3d/split_1:output:50up_sampling3d/split_1:output:50up_sampling3d/split_1:output:51up_sampling3d/split_1:output:51up_sampling3d/split_1:output:52up_sampling3d/split_1:output:52up_sampling3d/split_1:output:53up_sampling3d/split_1:output:53up_sampling3d/split_1:output:54up_sampling3d/split_1:output:54up_sampling3d/split_1:output:55up_sampling3d/split_1:output:55up_sampling3d/split_1:output:56up_sampling3d/split_1:output:56up_sampling3d/split_1:output:57up_sampling3d/split_1:output:57up_sampling3d/split_1:output:58up_sampling3d/split_1:output:58up_sampling3d/split_1:output:59up_sampling3d/split_1:output:59up_sampling3d/split_1:output:60up_sampling3d/split_1:output:60up_sampling3d/split_1:output:61up_sampling3d/split_1:output:61up_sampling3d/split_1:output:62up_sampling3d/split_1:output:62up_sampling3d/split_1:output:63up_sampling3d/split_1:output:63up_sampling3d/split_1:output:64up_sampling3d/split_1:output:64up_sampling3d/split_1:output:65up_sampling3d/split_1:output:65up_sampling3d/split_1:output:66up_sampling3d/split_1:output:66up_sampling3d/split_1:output:67up_sampling3d/split_1:output:67up_sampling3d/split_1:output:68up_sampling3d/split_1:output:68up_sampling3d/split_1:output:69up_sampling3d/split_1:output:69up_sampling3d/split_1:output:70up_sampling3d/split_1:output:70up_sampling3d/split_1:output:71up_sampling3d/split_1:output:71up_sampling3d/split_1:output:72up_sampling3d/split_1:output:72up_sampling3d/split_1:output:73up_sampling3d/split_1:output:73up_sampling3d/split_1:output:74up_sampling3d/split_1:output:74up_sampling3d/split_1:output:75up_sampling3d/split_1:output:75up_sampling3d/split_1:output:76up_sampling3d/split_1:output:76up_sampling3d/split_1:output:77up_sampling3d/split_1:output:77up_sampling3d/split_1:output:78up_sampling3d/split_1:output:78up_sampling3d/split_1:output:79up_sampling3d/split_1:output:79up_sampling3d/split_1:output:80up_sampling3d/split_1:output:80up_sampling3d/split_1:output:81up_sampling3d/split_1:output:81up_sampling3d/split_1:output:82up_sampling3d/split_1:output:82up_sampling3d/split_1:output:83up_sampling3d/split_1:output:83up_sampling3d/split_1:output:84up_sampling3d/split_1:output:84up_sampling3d/split_1:output:85up_sampling3d/split_1:output:85up_sampling3d/split_1:output:86up_sampling3d/split_1:output:86up_sampling3d/split_1:output:87up_sampling3d/split_1:output:87up_sampling3d/split_1:output:88up_sampling3d/split_1:output:88up_sampling3d/split_1:output:89up_sampling3d/split_1:output:89up_sampling3d/split_1:output:90up_sampling3d/split_1:output:90up_sampling3d/split_1:output:91up_sampling3d/split_1:output:91up_sampling3d/split_1:output:92up_sampling3d/split_1:output:92up_sampling3d/split_1:output:93up_sampling3d/split_1:output:93up_sampling3d/split_1:output:94up_sampling3d/split_1:output:94up_sampling3d/split_1:output:95up_sampling3d/split_1:output:95up_sampling3d/split_1:output:96up_sampling3d/split_1:output:96up_sampling3d/split_1:output:97up_sampling3d/split_1:output:97up_sampling3d/split_1:output:98up_sampling3d/split_1:output:98up_sampling3d/split_1:output:99up_sampling3d/split_1:output:99 up_sampling3d/split_1:output:100 up_sampling3d/split_1:output:100 up_sampling3d/split_1:output:101 up_sampling3d/split_1:output:101 up_sampling3d/split_1:output:102 up_sampling3d/split_1:output:102 up_sampling3d/split_1:output:103 up_sampling3d/split_1:output:103 up_sampling3d/split_1:output:104 up_sampling3d/split_1:output:104 up_sampling3d/split_1:output:105 up_sampling3d/split_1:output:105 up_sampling3d/split_1:output:106 up_sampling3d/split_1:output:106 up_sampling3d/split_1:output:107 up_sampling3d/split_1:output:107 up_sampling3d/split_1:output:108 up_sampling3d/split_1:output:108 up_sampling3d/split_1:output:109 up_sampling3d/split_1:output:109 up_sampling3d/split_1:output:110 up_sampling3d/split_1:output:110 up_sampling3d/split_1:output:111 up_sampling3d/split_1:output:111 up_sampling3d/split_1:output:112 up_sampling3d/split_1:output:112 up_sampling3d/split_1:output:113 up_sampling3d/split_1:output:113 up_sampling3d/split_1:output:114 up_sampling3d/split_1:output:114 up_sampling3d/split_1:output:115 up_sampling3d/split_1:output:115 up_sampling3d/split_1:output:116 up_sampling3d/split_1:output:116 up_sampling3d/split_1:output:117 up_sampling3d/split_1:output:117 up_sampling3d/split_1:output:118 up_sampling3d/split_1:output:118 up_sampling3d/split_1:output:119 up_sampling3d/split_1:output:119 up_sampling3d/split_1:output:120 up_sampling3d/split_1:output:120 up_sampling3d/split_1:output:121 up_sampling3d/split_1:output:121 up_sampling3d/split_1:output:122 up_sampling3d/split_1:output:122 up_sampling3d/split_1:output:123 up_sampling3d/split_1:output:123 up_sampling3d/split_1:output:124 up_sampling3d/split_1:output:124 up_sampling3d/split_1:output:125 up_sampling3d/split_1:output:125 up_sampling3d/split_1:output:126 up_sampling3d/split_1:output:126 up_sampling3d/split_1:output:127 up_sampling3d/split_1:output:127$up_sampling3d/concat_1/axis:output:0*
N*
T0*4
_output_shapes"
 :’’’’’’’’’@@a
up_sampling3d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
up_sampling3d/split_2Split(up_sampling3d/split_2/split_dim:output:0up_sampling3d/concat_1:output:0*
T0*
_output_shapes
:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@*
	num_split@]
up_sampling3d/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :ž!
up_sampling3d/concat_2ConcatV2up_sampling3d/split_2:output:0up_sampling3d/split_2:output:0up_sampling3d/split_2:output:1up_sampling3d/split_2:output:1up_sampling3d/split_2:output:2up_sampling3d/split_2:output:2up_sampling3d/split_2:output:3up_sampling3d/split_2:output:3up_sampling3d/split_2:output:4up_sampling3d/split_2:output:4up_sampling3d/split_2:output:5up_sampling3d/split_2:output:5up_sampling3d/split_2:output:6up_sampling3d/split_2:output:6up_sampling3d/split_2:output:7up_sampling3d/split_2:output:7up_sampling3d/split_2:output:8up_sampling3d/split_2:output:8up_sampling3d/split_2:output:9up_sampling3d/split_2:output:9up_sampling3d/split_2:output:10up_sampling3d/split_2:output:10up_sampling3d/split_2:output:11up_sampling3d/split_2:output:11up_sampling3d/split_2:output:12up_sampling3d/split_2:output:12up_sampling3d/split_2:output:13up_sampling3d/split_2:output:13up_sampling3d/split_2:output:14up_sampling3d/split_2:output:14up_sampling3d/split_2:output:15up_sampling3d/split_2:output:15up_sampling3d/split_2:output:16up_sampling3d/split_2:output:16up_sampling3d/split_2:output:17up_sampling3d/split_2:output:17up_sampling3d/split_2:output:18up_sampling3d/split_2:output:18up_sampling3d/split_2:output:19up_sampling3d/split_2:output:19up_sampling3d/split_2:output:20up_sampling3d/split_2:output:20up_sampling3d/split_2:output:21up_sampling3d/split_2:output:21up_sampling3d/split_2:output:22up_sampling3d/split_2:output:22up_sampling3d/split_2:output:23up_sampling3d/split_2:output:23up_sampling3d/split_2:output:24up_sampling3d/split_2:output:24up_sampling3d/split_2:output:25up_sampling3d/split_2:output:25up_sampling3d/split_2:output:26up_sampling3d/split_2:output:26up_sampling3d/split_2:output:27up_sampling3d/split_2:output:27up_sampling3d/split_2:output:28up_sampling3d/split_2:output:28up_sampling3d/split_2:output:29up_sampling3d/split_2:output:29up_sampling3d/split_2:output:30up_sampling3d/split_2:output:30up_sampling3d/split_2:output:31up_sampling3d/split_2:output:31up_sampling3d/split_2:output:32up_sampling3d/split_2:output:32up_sampling3d/split_2:output:33up_sampling3d/split_2:output:33up_sampling3d/split_2:output:34up_sampling3d/split_2:output:34up_sampling3d/split_2:output:35up_sampling3d/split_2:output:35up_sampling3d/split_2:output:36up_sampling3d/split_2:output:36up_sampling3d/split_2:output:37up_sampling3d/split_2:output:37up_sampling3d/split_2:output:38up_sampling3d/split_2:output:38up_sampling3d/split_2:output:39up_sampling3d/split_2:output:39up_sampling3d/split_2:output:40up_sampling3d/split_2:output:40up_sampling3d/split_2:output:41up_sampling3d/split_2:output:41up_sampling3d/split_2:output:42up_sampling3d/split_2:output:42up_sampling3d/split_2:output:43up_sampling3d/split_2:output:43up_sampling3d/split_2:output:44up_sampling3d/split_2:output:44up_sampling3d/split_2:output:45up_sampling3d/split_2:output:45up_sampling3d/split_2:output:46up_sampling3d/split_2:output:46up_sampling3d/split_2:output:47up_sampling3d/split_2:output:47up_sampling3d/split_2:output:48up_sampling3d/split_2:output:48up_sampling3d/split_2:output:49up_sampling3d/split_2:output:49up_sampling3d/split_2:output:50up_sampling3d/split_2:output:50up_sampling3d/split_2:output:51up_sampling3d/split_2:output:51up_sampling3d/split_2:output:52up_sampling3d/split_2:output:52up_sampling3d/split_2:output:53up_sampling3d/split_2:output:53up_sampling3d/split_2:output:54up_sampling3d/split_2:output:54up_sampling3d/split_2:output:55up_sampling3d/split_2:output:55up_sampling3d/split_2:output:56up_sampling3d/split_2:output:56up_sampling3d/split_2:output:57up_sampling3d/split_2:output:57up_sampling3d/split_2:output:58up_sampling3d/split_2:output:58up_sampling3d/split_2:output:59up_sampling3d/split_2:output:59up_sampling3d/split_2:output:60up_sampling3d/split_2:output:60up_sampling3d/split_2:output:61up_sampling3d/split_2:output:61up_sampling3d/split_2:output:62up_sampling3d/split_2:output:62up_sampling3d/split_2:output:63up_sampling3d/split_2:output:63$up_sampling3d/concat_2/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’@Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ķ
concatenate/concatConcatV2up_sampling3d/concat_2:output:0!conv_block3d/Relu_1:activations:0 concatenate/concat/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’`°
-conv_block3d_1/conv3d_2/Conv3D/ReadVariableOpReadVariableOp6conv_block3d_1_conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:` *
dtype0å
conv_block3d_1/conv3d_2/Conv3DConv3Dconcatenate/concat:output:05conv_block3d_1/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	
¢
.conv_block3d_1/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp7conv_block3d_1_conv3d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ė
conv_block3d_1/conv3d_2/BiasAddBiasAdd'conv_block3d_1/conv3d_2/Conv3D:output:06conv_block3d_1/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
conv_block3d_1/ReluRelu(conv_block3d_1/conv3d_2/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ °
-conv_block3d_1/conv3d_3/Conv3D/ReadVariableOpReadVariableOp6conv_block3d_1_conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0ė
conv_block3d_1/conv3d_3/Conv3DConv3D!conv_block3d_1/Relu:activations:05conv_block3d_1/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	
¢
.conv_block3d_1/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp7conv_block3d_1_conv3d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ė
conv_block3d_1/conv3d_3/BiasAddBiasAdd'conv_block3d_1/conv3d_3/Conv3D:output:06conv_block3d_1/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
conv_block3d_1/Relu_1Relu(conv_block3d_1/conv3d_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ ²
.conv_block3d_5/conv3d_10/Conv3D/ReadVariableOpReadVariableOp7conv_block3d_5_conv3d_10_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0ļ
conv_block3d_5/conv3d_10/Conv3DConv3D#conv_block3d_1/Relu_1:activations:06conv_block3d_5/conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’*
paddingSAME*
strides	
¤
/conv_block3d_5/conv3d_10/BiasAdd/ReadVariableOpReadVariableOp8conv_block3d_5_conv3d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ī
 conv_block3d_5/conv3d_10/BiasAddBiasAdd(conv_block3d_5/conv3d_10/Conv3D:output:07conv_block3d_5/conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’
IdentityIdentity)conv_block3d_5/conv3d_10/BiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ē
NoOpNoOp+^conv_block3d/conv3d/BiasAdd/ReadVariableOp*^conv_block3d/conv3d/Conv3D/ReadVariableOp-^conv_block3d/conv3d_1/BiasAdd/ReadVariableOp,^conv_block3d/conv3d_1/Conv3D/ReadVariableOp/^conv_block3d_1/conv3d_2/BiasAdd/ReadVariableOp.^conv_block3d_1/conv3d_2/Conv3D/ReadVariableOp/^conv_block3d_1/conv3d_3/BiasAdd/ReadVariableOp.^conv_block3d_1/conv3d_3/Conv3D/ReadVariableOp/^conv_block3d_2/conv3d_4/BiasAdd/ReadVariableOp.^conv_block3d_2/conv3d_4/Conv3D/ReadVariableOp/^conv_block3d_2/conv3d_5/BiasAdd/ReadVariableOp.^conv_block3d_2/conv3d_5/Conv3D/ReadVariableOp/^conv_block3d_3/conv3d_6/BiasAdd/ReadVariableOp.^conv_block3d_3/conv3d_6/Conv3D/ReadVariableOp/^conv_block3d_3/conv3d_7/BiasAdd/ReadVariableOp.^conv_block3d_3/conv3d_7/Conv3D/ReadVariableOp/^conv_block3d_4/conv3d_8/BiasAdd/ReadVariableOp.^conv_block3d_4/conv3d_8/Conv3D/ReadVariableOp/^conv_block3d_4/conv3d_9/BiasAdd/ReadVariableOp.^conv_block3d_4/conv3d_9/Conv3D/ReadVariableOp0^conv_block3d_5/conv3d_10/BiasAdd/ReadVariableOp/^conv_block3d_5/conv3d_10/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : 2X
*conv_block3d/conv3d/BiasAdd/ReadVariableOp*conv_block3d/conv3d/BiasAdd/ReadVariableOp2V
)conv_block3d/conv3d/Conv3D/ReadVariableOp)conv_block3d/conv3d/Conv3D/ReadVariableOp2\
,conv_block3d/conv3d_1/BiasAdd/ReadVariableOp,conv_block3d/conv3d_1/BiasAdd/ReadVariableOp2Z
+conv_block3d/conv3d_1/Conv3D/ReadVariableOp+conv_block3d/conv3d_1/Conv3D/ReadVariableOp2`
.conv_block3d_1/conv3d_2/BiasAdd/ReadVariableOp.conv_block3d_1/conv3d_2/BiasAdd/ReadVariableOp2^
-conv_block3d_1/conv3d_2/Conv3D/ReadVariableOp-conv_block3d_1/conv3d_2/Conv3D/ReadVariableOp2`
.conv_block3d_1/conv3d_3/BiasAdd/ReadVariableOp.conv_block3d_1/conv3d_3/BiasAdd/ReadVariableOp2^
-conv_block3d_1/conv3d_3/Conv3D/ReadVariableOp-conv_block3d_1/conv3d_3/Conv3D/ReadVariableOp2`
.conv_block3d_2/conv3d_4/BiasAdd/ReadVariableOp.conv_block3d_2/conv3d_4/BiasAdd/ReadVariableOp2^
-conv_block3d_2/conv3d_4/Conv3D/ReadVariableOp-conv_block3d_2/conv3d_4/Conv3D/ReadVariableOp2`
.conv_block3d_2/conv3d_5/BiasAdd/ReadVariableOp.conv_block3d_2/conv3d_5/BiasAdd/ReadVariableOp2^
-conv_block3d_2/conv3d_5/Conv3D/ReadVariableOp-conv_block3d_2/conv3d_5/Conv3D/ReadVariableOp2`
.conv_block3d_3/conv3d_6/BiasAdd/ReadVariableOp.conv_block3d_3/conv3d_6/BiasAdd/ReadVariableOp2^
-conv_block3d_3/conv3d_6/Conv3D/ReadVariableOp-conv_block3d_3/conv3d_6/Conv3D/ReadVariableOp2`
.conv_block3d_3/conv3d_7/BiasAdd/ReadVariableOp.conv_block3d_3/conv3d_7/BiasAdd/ReadVariableOp2^
-conv_block3d_3/conv3d_7/Conv3D/ReadVariableOp-conv_block3d_3/conv3d_7/Conv3D/ReadVariableOp2`
.conv_block3d_4/conv3d_8/BiasAdd/ReadVariableOp.conv_block3d_4/conv3d_8/BiasAdd/ReadVariableOp2^
-conv_block3d_4/conv3d_8/Conv3D/ReadVariableOp-conv_block3d_4/conv3d_8/Conv3D/ReadVariableOp2`
.conv_block3d_4/conv3d_9/BiasAdd/ReadVariableOp.conv_block3d_4/conv3d_9/BiasAdd/ReadVariableOp2^
-conv_block3d_4/conv3d_9/Conv3D/ReadVariableOp-conv_block3d_4/conv3d_9/Conv3D/ReadVariableOp2b
/conv_block3d_5/conv3d_10/BiasAdd/ReadVariableOp/conv_block3d_5/conv3d_10/BiasAdd/ReadVariableOp2`
.conv_block3d_5/conv3d_10/Conv3D/ReadVariableOp.conv_block3d_5/conv3d_10/Conv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’
 
_user_specified_nameinputs
Į
Č
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1041004

inputs,
conv3d_1040980: 
conv3d_1040982: .
conv3d_1_1040997:  
conv3d_1_1040999: 
identity¢conv3d/StatefulPartitionedCall¢ conv3d_1/StatefulPartitionedCallž
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_1040980conv3d_1040982*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_conv3d_layer_call_and_return_conditional_losses_1040979u
ReluRelu'conv3d/StatefulPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_1_1040997conv3d_1_1040999*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_1_layer_call_and_return_conditional_losses_1040996y
Relu_1Relu)conv3d_1/StatefulPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ q
IdentityIdentityRelu_1:activations:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ 
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’: : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’
 
_user_specified_nameinputs
×
f
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_1044019

inputs
identity½
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
ksize	
*
paddingSAME*
strides	

IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’: {
W
_output_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ł
ė
0__inference_conv_block3d_1_layer_call_fn_1043898

inputs%
unknown:` 
	unknown_0: '
	unknown_1:  
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1041484}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’`
 
_user_specified_nameinputs
ū
ń
0__inference_conv_block3d_4_layer_call_fn_1041410
input_1&
unknown:@
	unknown_0:	)
	unknown_1:
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’@ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1041386|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’@ @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:’’’’’’’’’@ @
!
_user_specified_name	input_1
Ń


E__inference_conv3d_1_layer_call_and_return_conditional_losses_1044451

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0 
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’ 
 
_user_specified_nameinputs

©
0__inference_conv_block3d_5_layer_call_fn_1043670

inputs%
unknown: 
	unknown_0:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1041810}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’ 
 
_user_specified_nameinputs
Õ
Ń
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1041706

inputs/
conv3d_6_1041693:Ą@
conv3d_6_1041695:@.
conv3d_7_1041699:@@
conv3d_7_1041701:@
identity¢ conv3d_6/StatefulPartitionedCall¢ conv3d_7/StatefulPartitionedCall
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_6_1041693conv3d_6_1041695*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_6_layer_call_and_return_conditional_losses_1041619v
ReluRelu)conv3d_6/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_7_1041699conv3d_7_1041701*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_7_layer_call_and_return_conditional_losses_1041636x
Relu_1Relu)conv3d_7/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@p
IdentityIdentityRelu_1:activations:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
NoOpNoOp!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’
@Ą: : : : 2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’
@Ą
 
_user_specified_nameinputs
ų
š
0__inference_conv_block3d_4_layer_call_fn_1043849

inputs&
unknown:@
	unknown_0:	)
	unknown_1:
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’@ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1041386|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’@ @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:’’’’’’’’’@ @
 
_user_specified_nameinputs
Ģ


E__inference_conv3d_4_layer_call_and_return_conditional_losses_1041139

inputs<
conv3d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: @*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’
@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:\ X
4
_output_shapes"
 :’’’’’’’’’
@ 
 
_user_specified_nameinputs
ū
ķ
0__inference_conv_block3d_3_layer_call_fn_1041730
input_1&
unknown:Ą@
	unknown_0:@'
	unknown_1:@@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1041706|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’
@Ą: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’
@Ą
!
_user_specified_name	input_1
Ł
Š
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1041484

inputs.
conv3d_2_1041460:` 
conv3d_2_1041462: .
conv3d_3_1041477:  
conv3d_3_1041479: 
identity¢ conv3d_2/StatefulPartitionedCall¢ conv3d_3/StatefulPartitionedCall
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_2_1041460conv3d_2_1041462*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_2_layer_call_and_return_conditional_losses_1041459w
ReluRelu)conv3d_2/StatefulPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_3_1041477conv3d_3_1041479*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_3_layer_call_and_return_conditional_losses_1041476y
Relu_1Relu)conv3d_3/StatefulPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ q
IdentityIdentityRelu_1:activations:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ 
NoOpNoOp!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’`: : : : 2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’`
 
_user_specified_nameinputs
Š


E__inference_conv3d_6_layer_call_and_return_conditional_losses_1041619

inputs=
conv3d_readvariableop_resource:Ą@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:Ą@*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’
@Ą: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’
@Ą
 
_user_specified_nameinputs
Ä
Ä
%__inference_signature_wrapper_1042749
input_1%
unknown: 
	unknown_0: '
	unknown_1:  
	unknown_2: '
	unknown_3: @
	unknown_4:@'
	unknown_5:@@
	unknown_6:@(
	unknown_7:@
	unknown_8:	)
	unknown_9:

unknown_10:	)

unknown_11:Ą@

unknown_12:@(

unknown_13:@@

unknown_14:@(

unknown_15:` 

unknown_16: (

unknown_17:  

unknown_18: (

unknown_19: 

unknown_20:
identity¢StatefulPartitionedCallŚ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *+
f&R$
"__inference__wrapped_model_1040962}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’
!
_user_specified_name	input_1
Ö
«%
 __inference__traced_save_1044845
file_prefix9
5savev2_conv_block3d_conv3d_kernel_read_readvariableop7
3savev2_conv_block3d_conv3d_bias_read_readvariableop;
7savev2_conv_block3d_conv3d_1_kernel_read_readvariableop9
5savev2_conv_block3d_conv3d_1_bias_read_readvariableop=
9savev2_conv_block3d_2_conv3d_4_kernel_read_readvariableop;
7savev2_conv_block3d_2_conv3d_4_bias_read_readvariableop=
9savev2_conv_block3d_2_conv3d_5_kernel_read_readvariableop;
7savev2_conv_block3d_2_conv3d_5_bias_read_readvariableop=
9savev2_conv_block3d_4_conv3d_8_kernel_read_readvariableop;
7savev2_conv_block3d_4_conv3d_8_bias_read_readvariableop=
9savev2_conv_block3d_4_conv3d_9_kernel_read_readvariableop;
7savev2_conv_block3d_4_conv3d_9_bias_read_readvariableop=
9savev2_conv_block3d_1_conv3d_2_kernel_read_readvariableop;
7savev2_conv_block3d_1_conv3d_2_bias_read_readvariableop=
9savev2_conv_block3d_1_conv3d_3_kernel_read_readvariableop;
7savev2_conv_block3d_1_conv3d_3_bias_read_readvariableop=
9savev2_conv_block3d_3_conv3d_6_kernel_read_readvariableop;
7savev2_conv_block3d_3_conv3d_6_bias_read_readvariableop=
9savev2_conv_block3d_3_conv3d_7_kernel_read_readvariableop;
7savev2_conv_block3d_3_conv3d_7_bias_read_readvariableop>
:savev2_conv_block3d_5_conv3d_10_kernel_read_readvariableop<
8savev2_conv_block3d_5_conv3d_10_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop@
<savev2_adam_conv_block3d_conv3d_kernel_m_read_readvariableop>
:savev2_adam_conv_block3d_conv3d_bias_m_read_readvariableopB
>savev2_adam_conv_block3d_conv3d_1_kernel_m_read_readvariableop@
<savev2_adam_conv_block3d_conv3d_1_bias_m_read_readvariableopD
@savev2_adam_conv_block3d_2_conv3d_4_kernel_m_read_readvariableopB
>savev2_adam_conv_block3d_2_conv3d_4_bias_m_read_readvariableopD
@savev2_adam_conv_block3d_2_conv3d_5_kernel_m_read_readvariableopB
>savev2_adam_conv_block3d_2_conv3d_5_bias_m_read_readvariableopD
@savev2_adam_conv_block3d_4_conv3d_8_kernel_m_read_readvariableopB
>savev2_adam_conv_block3d_4_conv3d_8_bias_m_read_readvariableopD
@savev2_adam_conv_block3d_4_conv3d_9_kernel_m_read_readvariableopB
>savev2_adam_conv_block3d_4_conv3d_9_bias_m_read_readvariableopD
@savev2_adam_conv_block3d_1_conv3d_2_kernel_m_read_readvariableopB
>savev2_adam_conv_block3d_1_conv3d_2_bias_m_read_readvariableopD
@savev2_adam_conv_block3d_1_conv3d_3_kernel_m_read_readvariableopB
>savev2_adam_conv_block3d_1_conv3d_3_bias_m_read_readvariableopD
@savev2_adam_conv_block3d_3_conv3d_6_kernel_m_read_readvariableopB
>savev2_adam_conv_block3d_3_conv3d_6_bias_m_read_readvariableopD
@savev2_adam_conv_block3d_3_conv3d_7_kernel_m_read_readvariableopB
>savev2_adam_conv_block3d_3_conv3d_7_bias_m_read_readvariableopE
Asavev2_adam_conv_block3d_5_conv3d_10_kernel_m_read_readvariableopC
?savev2_adam_conv_block3d_5_conv3d_10_bias_m_read_readvariableop@
<savev2_adam_conv_block3d_conv3d_kernel_v_read_readvariableop>
:savev2_adam_conv_block3d_conv3d_bias_v_read_readvariableopB
>savev2_adam_conv_block3d_conv3d_1_kernel_v_read_readvariableop@
<savev2_adam_conv_block3d_conv3d_1_bias_v_read_readvariableopD
@savev2_adam_conv_block3d_2_conv3d_4_kernel_v_read_readvariableopB
>savev2_adam_conv_block3d_2_conv3d_4_bias_v_read_readvariableopD
@savev2_adam_conv_block3d_2_conv3d_5_kernel_v_read_readvariableopB
>savev2_adam_conv_block3d_2_conv3d_5_bias_v_read_readvariableopD
@savev2_adam_conv_block3d_4_conv3d_8_kernel_v_read_readvariableopB
>savev2_adam_conv_block3d_4_conv3d_8_bias_v_read_readvariableopD
@savev2_adam_conv_block3d_4_conv3d_9_kernel_v_read_readvariableopB
>savev2_adam_conv_block3d_4_conv3d_9_bias_v_read_readvariableopD
@savev2_adam_conv_block3d_1_conv3d_2_kernel_v_read_readvariableopB
>savev2_adam_conv_block3d_1_conv3d_2_bias_v_read_readvariableopD
@savev2_adam_conv_block3d_1_conv3d_3_kernel_v_read_readvariableopB
>savev2_adam_conv_block3d_1_conv3d_3_bias_v_read_readvariableopD
@savev2_adam_conv_block3d_3_conv3d_6_kernel_v_read_readvariableopB
>savev2_adam_conv_block3d_3_conv3d_6_bias_v_read_readvariableopD
@savev2_adam_conv_block3d_3_conv3d_7_kernel_v_read_readvariableopB
>savev2_adam_conv_block3d_3_conv3d_7_bias_v_read_readvariableopE
Asavev2_adam_conv_block3d_5_conv3d_10_kernel_v_read_readvariableopC
?savev2_adam_conv_block3d_5_conv3d_10_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: "
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*Č!
value¾!B»!JB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*©
valueBJB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B $
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_conv_block3d_conv3d_kernel_read_readvariableop3savev2_conv_block3d_conv3d_bias_read_readvariableop7savev2_conv_block3d_conv3d_1_kernel_read_readvariableop5savev2_conv_block3d_conv3d_1_bias_read_readvariableop9savev2_conv_block3d_2_conv3d_4_kernel_read_readvariableop7savev2_conv_block3d_2_conv3d_4_bias_read_readvariableop9savev2_conv_block3d_2_conv3d_5_kernel_read_readvariableop7savev2_conv_block3d_2_conv3d_5_bias_read_readvariableop9savev2_conv_block3d_4_conv3d_8_kernel_read_readvariableop7savev2_conv_block3d_4_conv3d_8_bias_read_readvariableop9savev2_conv_block3d_4_conv3d_9_kernel_read_readvariableop7savev2_conv_block3d_4_conv3d_9_bias_read_readvariableop9savev2_conv_block3d_1_conv3d_2_kernel_read_readvariableop7savev2_conv_block3d_1_conv3d_2_bias_read_readvariableop9savev2_conv_block3d_1_conv3d_3_kernel_read_readvariableop7savev2_conv_block3d_1_conv3d_3_bias_read_readvariableop9savev2_conv_block3d_3_conv3d_6_kernel_read_readvariableop7savev2_conv_block3d_3_conv3d_6_bias_read_readvariableop9savev2_conv_block3d_3_conv3d_7_kernel_read_readvariableop7savev2_conv_block3d_3_conv3d_7_bias_read_readvariableop:savev2_conv_block3d_5_conv3d_10_kernel_read_readvariableop8savev2_conv_block3d_5_conv3d_10_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop<savev2_adam_conv_block3d_conv3d_kernel_m_read_readvariableop:savev2_adam_conv_block3d_conv3d_bias_m_read_readvariableop>savev2_adam_conv_block3d_conv3d_1_kernel_m_read_readvariableop<savev2_adam_conv_block3d_conv3d_1_bias_m_read_readvariableop@savev2_adam_conv_block3d_2_conv3d_4_kernel_m_read_readvariableop>savev2_adam_conv_block3d_2_conv3d_4_bias_m_read_readvariableop@savev2_adam_conv_block3d_2_conv3d_5_kernel_m_read_readvariableop>savev2_adam_conv_block3d_2_conv3d_5_bias_m_read_readvariableop@savev2_adam_conv_block3d_4_conv3d_8_kernel_m_read_readvariableop>savev2_adam_conv_block3d_4_conv3d_8_bias_m_read_readvariableop@savev2_adam_conv_block3d_4_conv3d_9_kernel_m_read_readvariableop>savev2_adam_conv_block3d_4_conv3d_9_bias_m_read_readvariableop@savev2_adam_conv_block3d_1_conv3d_2_kernel_m_read_readvariableop>savev2_adam_conv_block3d_1_conv3d_2_bias_m_read_readvariableop@savev2_adam_conv_block3d_1_conv3d_3_kernel_m_read_readvariableop>savev2_adam_conv_block3d_1_conv3d_3_bias_m_read_readvariableop@savev2_adam_conv_block3d_3_conv3d_6_kernel_m_read_readvariableop>savev2_adam_conv_block3d_3_conv3d_6_bias_m_read_readvariableop@savev2_adam_conv_block3d_3_conv3d_7_kernel_m_read_readvariableop>savev2_adam_conv_block3d_3_conv3d_7_bias_m_read_readvariableopAsavev2_adam_conv_block3d_5_conv3d_10_kernel_m_read_readvariableop?savev2_adam_conv_block3d_5_conv3d_10_bias_m_read_readvariableop<savev2_adam_conv_block3d_conv3d_kernel_v_read_readvariableop:savev2_adam_conv_block3d_conv3d_bias_v_read_readvariableop>savev2_adam_conv_block3d_conv3d_1_kernel_v_read_readvariableop<savev2_adam_conv_block3d_conv3d_1_bias_v_read_readvariableop@savev2_adam_conv_block3d_2_conv3d_4_kernel_v_read_readvariableop>savev2_adam_conv_block3d_2_conv3d_4_bias_v_read_readvariableop@savev2_adam_conv_block3d_2_conv3d_5_kernel_v_read_readvariableop>savev2_adam_conv_block3d_2_conv3d_5_bias_v_read_readvariableop@savev2_adam_conv_block3d_4_conv3d_8_kernel_v_read_readvariableop>savev2_adam_conv_block3d_4_conv3d_8_bias_v_read_readvariableop@savev2_adam_conv_block3d_4_conv3d_9_kernel_v_read_readvariableop>savev2_adam_conv_block3d_4_conv3d_9_bias_v_read_readvariableop@savev2_adam_conv_block3d_1_conv3d_2_kernel_v_read_readvariableop>savev2_adam_conv_block3d_1_conv3d_2_bias_v_read_readvariableop@savev2_adam_conv_block3d_1_conv3d_3_kernel_v_read_readvariableop>savev2_adam_conv_block3d_1_conv3d_3_bias_v_read_readvariableop@savev2_adam_conv_block3d_3_conv3d_6_kernel_v_read_readvariableop>savev2_adam_conv_block3d_3_conv3d_6_bias_v_read_readvariableop@savev2_adam_conv_block3d_3_conv3d_7_kernel_v_read_readvariableop>savev2_adam_conv_block3d_3_conv3d_7_bias_v_read_readvariableopAsavev2_adam_conv_block3d_5_conv3d_10_kernel_v_read_readvariableop?savev2_adam_conv_block3d_5_conv3d_10_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *X
dtypesN
L2J	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Õ
_input_shapesĆ
Ą: : : :  : : @:@:@@:@:@::::` : :  : :Ą@:@:@@:@: :: : : : : : : : : :  : : @:@:@@:@:@::::` : :  : :Ą@:@:@@:@: :: : :  : : @:@:@@:@:@::::` : :  : :Ą@:@:@@:@: :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
: : 

_output_shapes
: :0,
*
_output_shapes
:  : 

_output_shapes
: :0,
*
_output_shapes
: @: 

_output_shapes
:@:0,
*
_output_shapes
:@@: 

_output_shapes
:@:1	-
+
_output_shapes
:@:!


_output_shapes	
::2.
,
_output_shapes
::!

_output_shapes	
::0,
*
_output_shapes
:` : 

_output_shapes
: :0,
*
_output_shapes
:  : 

_output_shapes
: :1-
+
_output_shapes
:Ą@: 

_output_shapes
:@:0,
*
_output_shapes
:@@: 

_output_shapes
:@:0,
*
_output_shapes
: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :0,
*
_output_shapes
: : 

_output_shapes
: :0 ,
*
_output_shapes
:  : !

_output_shapes
: :0",
*
_output_shapes
: @: #

_output_shapes
:@:0$,
*
_output_shapes
:@@: %

_output_shapes
:@:1&-
+
_output_shapes
:@:!'

_output_shapes	
::2(.
,
_output_shapes
::!)

_output_shapes	
::0*,
*
_output_shapes
:` : +

_output_shapes
: :0,,
*
_output_shapes
:  : -

_output_shapes
: :1.-
+
_output_shapes
:Ą@: /

_output_shapes
:@:00,
*
_output_shapes
:@@: 1

_output_shapes
:@:02,
*
_output_shapes
: : 3

_output_shapes
::04,
*
_output_shapes
: : 5

_output_shapes
: :06,
*
_output_shapes
:  : 7

_output_shapes
: :08,
*
_output_shapes
: @: 9

_output_shapes
:@:0:,
*
_output_shapes
:@@: ;

_output_shapes
:@:1<-
+
_output_shapes
:@:!=

_output_shapes	
::2>.
,
_output_shapes
::!?

_output_shapes	
::0@,
*
_output_shapes
:` : A

_output_shapes
: :0B,
*
_output_shapes
:  : C

_output_shapes
: :1D-
+
_output_shapes
:Ą@: E

_output_shapes
:@:0F,
*
_output_shapes
:@@: G

_output_shapes
:@:0H,
*
_output_shapes
: : I

_output_shapes
::J

_output_shapes
: 

£
*__inference_conv3d_7_layer_call_fn_1044593

inputs%
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallģ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_7_layer_call_and_return_conditional_losses_1041636|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’
@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :’’’’’’’’’
@@
 
_user_specified_nameinputs
Ń


E__inference_conv3d_3_layer_call_and_return_conditional_losses_1044565

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0 
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’ 
 
_user_specified_nameinputs
õ
é
.__inference_conv_block3d_layer_call_fn_1043725

inputs%
unknown: 
	unknown_0: '
	unknown_1:  
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1041066}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’
 
_user_specified_nameinputs
ē
Ē
)__inference_u_net3d_layer_call_fn_1042847

inputs%
unknown: 
	unknown_0: '
	unknown_1:  
	unknown_2: '
	unknown_3: @
	unknown_4:@'
	unknown_5:@@
	unknown_6:@(
	unknown_7:@
	unknown_8:	)
	unknown_9:

unknown_10:	)

unknown_11:Ą@

unknown_12:@(

unknown_13:@@

unknown_14:@(

unknown_15:` 

unknown_16: (

unknown_17:  

unknown_18: (

unknown_19: 

unknown_20:
identity¢StatefulPartitionedCallū
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_u_net3d_layer_call_and_return_conditional_losses_1042476}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’
 
_user_specified_nameinputs
Ī=
Ź

D__inference_u_net3d_layer_call_and_return_conditional_losses_1042632
input_12
conv_block3d_1042575: "
conv_block3d_1042577: 2
conv_block3d_1042579:  "
conv_block3d_1042581: 4
conv_block3d_2_1042585: @$
conv_block3d_2_1042587:@4
conv_block3d_2_1042589:@@$
conv_block3d_2_1042591:@5
conv_block3d_4_1042595:@%
conv_block3d_4_1042597:	6
conv_block3d_4_1042599:%
conv_block3d_4_1042601:	5
conv_block3d_3_1042606:Ą@$
conv_block3d_3_1042608:@4
conv_block3d_3_1042610:@@$
conv_block3d_3_1042612:@4
conv_block3d_1_1042617:` $
conv_block3d_1_1042619: 4
conv_block3d_1_1042621:  $
conv_block3d_1_1042623: 4
conv_block3d_5_1042626: $
conv_block3d_5_1042628:
identity¢$conv_block3d/StatefulPartitionedCall¢&conv_block3d_1/StatefulPartitionedCall¢&conv_block3d_2/StatefulPartitionedCall¢&conv_block3d_3/StatefulPartitionedCall¢&conv_block3d_4/StatefulPartitionedCall¢&conv_block3d_5/StatefulPartitionedCallĒ
$conv_block3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv_block3d_1042575conv_block3d_1042577conv_block3d_1042579conv_block3d_1042581*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1041004ü
max_pooling3d/PartitionedCallPartitionedCall-conv_block3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_1041771ń
&conv_block3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv_block3d_2_1042585conv_block3d_2_1042587conv_block3d_2_1042589conv_block3d_2_1042591*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1041164
max_pooling3d_1/PartitionedCallPartitionedCall/conv_block3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:’’’’’’’’’@ @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_1041783ó
&conv_block3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv_block3d_4_1042595conv_block3d_4_1042597conv_block3d_4_1042599conv_block3d_4_1042601*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’@ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1041324
up_sampling3d_1/PartitionedCallPartitionedCall/conv_block3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’
@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_1042031Ŗ
concatenate_1/PartitionedCallPartitionedCall(up_sampling3d_1/PartitionedCall:output:0/conv_block3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’
@Ą* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1042040ń
&conv_block3d_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv_block3d_3_1042606conv_block3d_3_1042608conv_block3d_3_1042610conv_block3d_3_1042612*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1041644’
up_sampling3d/PartitionedCallPartitionedCall/conv_block3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_1042266¢
concatenate/PartitionedCallPartitionedCall&up_sampling3d/PartitionedCall:output:0-conv_block3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1042275š
&conv_block3d_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv_block3d_1_1042617conv_block3d_1_1042619conv_block3d_1_1042621conv_block3d_1_1042623*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1041484Ē
&conv_block3d_5/StatefulPartitionedCallStatefulPartitionedCall/conv_block3d_1/StatefulPartitionedCall:output:0conv_block3d_5_1042626conv_block3d_5_1042628*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1041810
IdentityIdentity/conv_block3d_5/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ŗ
NoOpNoOp%^conv_block3d/StatefulPartitionedCall'^conv_block3d_1/StatefulPartitionedCall'^conv_block3d_2/StatefulPartitionedCall'^conv_block3d_3/StatefulPartitionedCall'^conv_block3d_4/StatefulPartitionedCall'^conv_block3d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : 2L
$conv_block3d/StatefulPartitionedCall$conv_block3d/StatefulPartitionedCall2P
&conv_block3d_1/StatefulPartitionedCall&conv_block3d_1/StatefulPartitionedCall2P
&conv_block3d_2/StatefulPartitionedCall&conv_block3d_2/StatefulPartitionedCall2P
&conv_block3d_3/StatefulPartitionedCall&conv_block3d_3/StatefulPartitionedCall2P
&conv_block3d_4/StatefulPartitionedCall&conv_block3d_4/StatefulPartitionedCall2P
&conv_block3d_5/StatefulPartitionedCall&conv_block3d_5/StatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’
!
_user_specified_name	input_1

Ŗ
0__inference_conv_block3d_5_layer_call_fn_1041863
input_1%
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallō
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1041847}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’ 
!
_user_specified_name	input_1
Ģ


E__inference_conv3d_5_layer_call_and_return_conditional_losses_1041156

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’
@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:\ X
4
_output_shapes"
 :’’’’’’’’’
@@
 
_user_specified_nameinputs
Į
Č
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1041066

inputs,
conv3d_1041053: 
conv3d_1041055: .
conv3d_1_1041059:  
conv3d_1_1041061: 
identity¢conv3d/StatefulPartitionedCall¢ conv3d_1/StatefulPartitionedCallž
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_1041053conv3d_1041055*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_conv3d_layer_call_and_return_conditional_losses_1040979u
ReluRelu'conv3d/StatefulPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_1_1041059conv3d_1_1041061*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_1_layer_call_and_return_conditional_losses_1040996y
Relu_1Relu)conv3d_1/StatefulPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ q
IdentityIdentityRelu_1:activations:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ 
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’: : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’
 
_user_specified_nameinputs
Ń


E__inference_conv3d_1_layer_call_and_return_conditional_losses_1040996

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0 
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’ 
 
_user_specified_nameinputs
Ų
Ņ
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1041762
input_1/
conv3d_6_1041749:Ą@
conv3d_6_1041751:@.
conv3d_7_1041755:@@
conv3d_7_1041757:@
identity¢ conv3d_6/StatefulPartitionedCall¢ conv3d_7/StatefulPartitionedCall
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_6_1041749conv3d_6_1041751*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_6_layer_call_and_return_conditional_losses_1041619v
ReluRelu)conv3d_6/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_7_1041755conv3d_7_1041757*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_7_layer_call_and_return_conditional_losses_1041636x
Relu_1Relu)conv3d_7/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@p
IdentityIdentityRelu_1:activations:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
NoOpNoOp!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’
@Ą: : : : 2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’
@Ą
!
_user_specified_name	input_1
Õ
Õ
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1041324

inputs/
conv3d_8_1041300:@
conv3d_8_1041302:	0
conv3d_9_1041317:
conv3d_9_1041319:	
identity¢ conv3d_8/StatefulPartitionedCall¢ conv3d_9/StatefulPartitionedCall
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_8_1041300conv3d_8_1041302*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_8_layer_call_and_return_conditional_losses_1041299v
ReluRelu)conv3d_8/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ 
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_9_1041317conv3d_9_1041319*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_9_layer_call_and_return_conditional_losses_1041316x
Relu_1Relu)conv3d_9/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ p
IdentityIdentityRelu_1:activations:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’@ 
NoOpNoOp!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’@ @: : : : 2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall:[ W
3
_output_shapes!
:’’’’’’’’’@ @
 
_user_specified_nameinputs

v
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1044394
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’
@Ąe
IdentityIdentityconcat:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’
@Ą"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:’’’’’’’’’
@:’’’’’’’’’
@@:_ [
5
_output_shapes#
!:’’’’’’’’’
@
"
_user_specified_name
inputs/0:^Z
4
_output_shapes"
 :’’’’’’’’’
@@
"
_user_specified_name
inputs/1
	
į
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1041872
input_1/
conv3d_10_1041866: 
conv3d_10_1041868:
identity¢!conv3d_10/StatefulPartitionedCall
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_10_1041866conv3d_10_1041868*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv3d_10_layer_call_and_return_conditional_losses_1041803
IdentityIdentity*conv3d_10/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’j
NoOpNoOp"^conv3d_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’ 
!
_user_specified_name	input_1
āĮ
ž
"__inference__wrapped_model_1040962
input_1X
:u_net3d_conv_block3d_conv3d_conv3d_readvariableop_resource: I
;u_net3d_conv_block3d_conv3d_biasadd_readvariableop_resource: Z
<u_net3d_conv_block3d_conv3d_1_conv3d_readvariableop_resource:  K
=u_net3d_conv_block3d_conv3d_1_biasadd_readvariableop_resource: \
>u_net3d_conv_block3d_2_conv3d_4_conv3d_readvariableop_resource: @M
?u_net3d_conv_block3d_2_conv3d_4_biasadd_readvariableop_resource:@\
>u_net3d_conv_block3d_2_conv3d_5_conv3d_readvariableop_resource:@@M
?u_net3d_conv_block3d_2_conv3d_5_biasadd_readvariableop_resource:@]
>u_net3d_conv_block3d_4_conv3d_8_conv3d_readvariableop_resource:@N
?u_net3d_conv_block3d_4_conv3d_8_biasadd_readvariableop_resource:	^
>u_net3d_conv_block3d_4_conv3d_9_conv3d_readvariableop_resource:N
?u_net3d_conv_block3d_4_conv3d_9_biasadd_readvariableop_resource:	]
>u_net3d_conv_block3d_3_conv3d_6_conv3d_readvariableop_resource:Ą@M
?u_net3d_conv_block3d_3_conv3d_6_biasadd_readvariableop_resource:@\
>u_net3d_conv_block3d_3_conv3d_7_conv3d_readvariableop_resource:@@M
?u_net3d_conv_block3d_3_conv3d_7_biasadd_readvariableop_resource:@\
>u_net3d_conv_block3d_1_conv3d_2_conv3d_readvariableop_resource:` M
?u_net3d_conv_block3d_1_conv3d_2_biasadd_readvariableop_resource: \
>u_net3d_conv_block3d_1_conv3d_3_conv3d_readvariableop_resource:  M
?u_net3d_conv_block3d_1_conv3d_3_biasadd_readvariableop_resource: ]
?u_net3d_conv_block3d_5_conv3d_10_conv3d_readvariableop_resource: N
@u_net3d_conv_block3d_5_conv3d_10_biasadd_readvariableop_resource:
identity¢2u_net3d/conv_block3d/conv3d/BiasAdd/ReadVariableOp¢1u_net3d/conv_block3d/conv3d/Conv3D/ReadVariableOp¢4u_net3d/conv_block3d/conv3d_1/BiasAdd/ReadVariableOp¢3u_net3d/conv_block3d/conv3d_1/Conv3D/ReadVariableOp¢6u_net3d/conv_block3d_1/conv3d_2/BiasAdd/ReadVariableOp¢5u_net3d/conv_block3d_1/conv3d_2/Conv3D/ReadVariableOp¢6u_net3d/conv_block3d_1/conv3d_3/BiasAdd/ReadVariableOp¢5u_net3d/conv_block3d_1/conv3d_3/Conv3D/ReadVariableOp¢6u_net3d/conv_block3d_2/conv3d_4/BiasAdd/ReadVariableOp¢5u_net3d/conv_block3d_2/conv3d_4/Conv3D/ReadVariableOp¢6u_net3d/conv_block3d_2/conv3d_5/BiasAdd/ReadVariableOp¢5u_net3d/conv_block3d_2/conv3d_5/Conv3D/ReadVariableOp¢6u_net3d/conv_block3d_3/conv3d_6/BiasAdd/ReadVariableOp¢5u_net3d/conv_block3d_3/conv3d_6/Conv3D/ReadVariableOp¢6u_net3d/conv_block3d_3/conv3d_7/BiasAdd/ReadVariableOp¢5u_net3d/conv_block3d_3/conv3d_7/Conv3D/ReadVariableOp¢6u_net3d/conv_block3d_4/conv3d_8/BiasAdd/ReadVariableOp¢5u_net3d/conv_block3d_4/conv3d_8/Conv3D/ReadVariableOp¢6u_net3d/conv_block3d_4/conv3d_9/BiasAdd/ReadVariableOp¢5u_net3d/conv_block3d_4/conv3d_9/Conv3D/ReadVariableOp¢7u_net3d/conv_block3d_5/conv3d_10/BiasAdd/ReadVariableOp¢6u_net3d/conv_block3d_5/conv3d_10/Conv3D/ReadVariableOpø
1u_net3d/conv_block3d/conv3d/Conv3D/ReadVariableOpReadVariableOp:u_net3d_conv_block3d_conv3d_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0Ł
"u_net3d/conv_block3d/conv3d/Conv3DConv3Dinput_19u_net3d/conv_block3d/conv3d/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	
Ŗ
2u_net3d/conv_block3d/conv3d/BiasAdd/ReadVariableOpReadVariableOp;u_net3d_conv_block3d_conv3d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0×
#u_net3d/conv_block3d/conv3d/BiasAddBiasAdd+u_net3d/conv_block3d/conv3d/Conv3D:output:0:u_net3d/conv_block3d/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
u_net3d/conv_block3d/ReluRelu,u_net3d/conv_block3d/conv3d/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ ¼
3u_net3d/conv_block3d/conv3d_1/Conv3D/ReadVariableOpReadVariableOp<u_net3d_conv_block3d_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0ż
$u_net3d/conv_block3d/conv3d_1/Conv3DConv3D'u_net3d/conv_block3d/Relu:activations:0;u_net3d/conv_block3d/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	
®
4u_net3d/conv_block3d/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp=u_net3d_conv_block3d_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ż
%u_net3d/conv_block3d/conv3d_1/BiasAddBiasAdd-u_net3d/conv_block3d/conv3d_1/Conv3D:output:0<u_net3d/conv_block3d/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
u_net3d/conv_block3d/Relu_1Relu.u_net3d/conv_block3d/conv3d_1/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ Ó
u_net3d/max_pooling3d/MaxPool3D	MaxPool3D)u_net3d/conv_block3d/Relu_1:activations:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@ *
ksize	
*
paddingSAME*
strides	
Ą
5u_net3d/conv_block3d_2/conv3d_4/Conv3D/ReadVariableOpReadVariableOp>u_net3d_conv_block3d_2_conv3d_4_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0
&u_net3d/conv_block3d_2/conv3d_4/Conv3DConv3D(u_net3d/max_pooling3d/MaxPool3D:output:0=u_net3d/conv_block3d_2/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
²
6u_net3d/conv_block3d_2/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp?u_net3d_conv_block3d_2_conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ā
'u_net3d/conv_block3d_2/conv3d_4/BiasAddBiasAdd/u_net3d/conv_block3d_2/conv3d_4/Conv3D:output:0>u_net3d/conv_block3d_2/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
u_net3d/conv_block3d_2/ReluRelu0u_net3d/conv_block3d_2/conv3d_4/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@Ą
5u_net3d/conv_block3d_2/conv3d_5/Conv3D/ReadVariableOpReadVariableOp>u_net3d_conv_block3d_2_conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0
&u_net3d/conv_block3d_2/conv3d_5/Conv3DConv3D)u_net3d/conv_block3d_2/Relu:activations:0=u_net3d/conv_block3d_2/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
²
6u_net3d/conv_block3d_2/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp?u_net3d_conv_block3d_2_conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ā
'u_net3d/conv_block3d_2/conv3d_5/BiasAddBiasAdd/u_net3d/conv_block3d_2/conv3d_5/Conv3D:output:0>u_net3d/conv_block3d_2/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
u_net3d/conv_block3d_2/Relu_1Relu0u_net3d/conv_block3d_2/conv3d_5/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@Ö
!u_net3d/max_pooling3d_1/MaxPool3D	MaxPool3D+u_net3d/conv_block3d_2/Relu_1:activations:0*
T0*3
_output_shapes!
:’’’’’’’’’@ @*
ksize	
*
paddingSAME*
strides	
Į
5u_net3d/conv_block3d_4/conv3d_8/Conv3D/ReadVariableOpReadVariableOp>u_net3d_conv_block3d_4_conv3d_8_conv3d_readvariableop_resource*+
_output_shapes
:@*
dtype0
&u_net3d/conv_block3d_4/conv3d_8/Conv3DConv3D*u_net3d/max_pooling3d_1/MaxPool3D:output:0=u_net3d/conv_block3d_4/conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ *
paddingSAME*
strides	
³
6u_net3d/conv_block3d_4/conv3d_8/BiasAdd/ReadVariableOpReadVariableOp?u_net3d_conv_block3d_4_conv3d_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ā
'u_net3d/conv_block3d_4/conv3d_8/BiasAddBiasAdd/u_net3d/conv_block3d_4/conv3d_8/Conv3D:output:0>u_net3d/conv_block3d_4/conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ 
u_net3d/conv_block3d_4/ReluRelu0u_net3d/conv_block3d_4/conv3d_8/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ Ā
5u_net3d/conv_block3d_4/conv3d_9/Conv3D/ReadVariableOpReadVariableOp>u_net3d_conv_block3d_4_conv3d_9_conv3d_readvariableop_resource*,
_output_shapes
:*
dtype0
&u_net3d/conv_block3d_4/conv3d_9/Conv3DConv3D)u_net3d/conv_block3d_4/Relu:activations:0=u_net3d/conv_block3d_4/conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ *
paddingSAME*
strides	
³
6u_net3d/conv_block3d_4/conv3d_9/BiasAdd/ReadVariableOpReadVariableOp?u_net3d_conv_block3d_4_conv3d_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ā
'u_net3d/conv_block3d_4/conv3d_9/BiasAddBiasAdd/u_net3d/conv_block3d_4/conv3d_9/Conv3D:output:0>u_net3d/conv_block3d_4/conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ 
u_net3d/conv_block3d_4/Relu_1Relu0u_net3d/conv_block3d_4/conv3d_9/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ i
'u_net3d/up_sampling3d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ų
u_net3d/up_sampling3d_1/splitSplit0u_net3d/up_sampling3d_1/split/split_dim:output:0+u_net3d/conv_block3d_4/Relu_1:activations:0*
T0*¶
_output_shapes£
 :’’’’’’’’’@ :’’’’’’’’’@ :’’’’’’’’’@ :’’’’’’’’’@ :’’’’’’’’’@ *
	num_splite
#u_net3d/up_sampling3d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :°
u_net3d/up_sampling3d_1/concatConcatV2&u_net3d/up_sampling3d_1/split:output:0&u_net3d/up_sampling3d_1/split:output:0&u_net3d/up_sampling3d_1/split:output:1&u_net3d/up_sampling3d_1/split:output:1&u_net3d/up_sampling3d_1/split:output:2&u_net3d/up_sampling3d_1/split:output:2&u_net3d/up_sampling3d_1/split:output:3&u_net3d/up_sampling3d_1/split:output:3&u_net3d/up_sampling3d_1/split:output:4&u_net3d/up_sampling3d_1/split:output:4,u_net3d/up_sampling3d_1/concat/axis:output:0*
N
*
T0*4
_output_shapes"
 :’’’’’’’’’
@ k
)u_net3d/up_sampling3d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ø
u_net3d/up_sampling3d_1/split_1Split2u_net3d/up_sampling3d_1/split_1/split_dim:output:0'u_net3d/up_sampling3d_1/concat:output:0*
T0*
_output_shapes
:’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 *
	num_split@g
%u_net3d/up_sampling3d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :,
 u_net3d/up_sampling3d_1/concat_1ConcatV2(u_net3d/up_sampling3d_1/split_1:output:0(u_net3d/up_sampling3d_1/split_1:output:0(u_net3d/up_sampling3d_1/split_1:output:1(u_net3d/up_sampling3d_1/split_1:output:1(u_net3d/up_sampling3d_1/split_1:output:2(u_net3d/up_sampling3d_1/split_1:output:2(u_net3d/up_sampling3d_1/split_1:output:3(u_net3d/up_sampling3d_1/split_1:output:3(u_net3d/up_sampling3d_1/split_1:output:4(u_net3d/up_sampling3d_1/split_1:output:4(u_net3d/up_sampling3d_1/split_1:output:5(u_net3d/up_sampling3d_1/split_1:output:5(u_net3d/up_sampling3d_1/split_1:output:6(u_net3d/up_sampling3d_1/split_1:output:6(u_net3d/up_sampling3d_1/split_1:output:7(u_net3d/up_sampling3d_1/split_1:output:7(u_net3d/up_sampling3d_1/split_1:output:8(u_net3d/up_sampling3d_1/split_1:output:8(u_net3d/up_sampling3d_1/split_1:output:9(u_net3d/up_sampling3d_1/split_1:output:9)u_net3d/up_sampling3d_1/split_1:output:10)u_net3d/up_sampling3d_1/split_1:output:10)u_net3d/up_sampling3d_1/split_1:output:11)u_net3d/up_sampling3d_1/split_1:output:11)u_net3d/up_sampling3d_1/split_1:output:12)u_net3d/up_sampling3d_1/split_1:output:12)u_net3d/up_sampling3d_1/split_1:output:13)u_net3d/up_sampling3d_1/split_1:output:13)u_net3d/up_sampling3d_1/split_1:output:14)u_net3d/up_sampling3d_1/split_1:output:14)u_net3d/up_sampling3d_1/split_1:output:15)u_net3d/up_sampling3d_1/split_1:output:15)u_net3d/up_sampling3d_1/split_1:output:16)u_net3d/up_sampling3d_1/split_1:output:16)u_net3d/up_sampling3d_1/split_1:output:17)u_net3d/up_sampling3d_1/split_1:output:17)u_net3d/up_sampling3d_1/split_1:output:18)u_net3d/up_sampling3d_1/split_1:output:18)u_net3d/up_sampling3d_1/split_1:output:19)u_net3d/up_sampling3d_1/split_1:output:19)u_net3d/up_sampling3d_1/split_1:output:20)u_net3d/up_sampling3d_1/split_1:output:20)u_net3d/up_sampling3d_1/split_1:output:21)u_net3d/up_sampling3d_1/split_1:output:21)u_net3d/up_sampling3d_1/split_1:output:22)u_net3d/up_sampling3d_1/split_1:output:22)u_net3d/up_sampling3d_1/split_1:output:23)u_net3d/up_sampling3d_1/split_1:output:23)u_net3d/up_sampling3d_1/split_1:output:24)u_net3d/up_sampling3d_1/split_1:output:24)u_net3d/up_sampling3d_1/split_1:output:25)u_net3d/up_sampling3d_1/split_1:output:25)u_net3d/up_sampling3d_1/split_1:output:26)u_net3d/up_sampling3d_1/split_1:output:26)u_net3d/up_sampling3d_1/split_1:output:27)u_net3d/up_sampling3d_1/split_1:output:27)u_net3d/up_sampling3d_1/split_1:output:28)u_net3d/up_sampling3d_1/split_1:output:28)u_net3d/up_sampling3d_1/split_1:output:29)u_net3d/up_sampling3d_1/split_1:output:29)u_net3d/up_sampling3d_1/split_1:output:30)u_net3d/up_sampling3d_1/split_1:output:30)u_net3d/up_sampling3d_1/split_1:output:31)u_net3d/up_sampling3d_1/split_1:output:31)u_net3d/up_sampling3d_1/split_1:output:32)u_net3d/up_sampling3d_1/split_1:output:32)u_net3d/up_sampling3d_1/split_1:output:33)u_net3d/up_sampling3d_1/split_1:output:33)u_net3d/up_sampling3d_1/split_1:output:34)u_net3d/up_sampling3d_1/split_1:output:34)u_net3d/up_sampling3d_1/split_1:output:35)u_net3d/up_sampling3d_1/split_1:output:35)u_net3d/up_sampling3d_1/split_1:output:36)u_net3d/up_sampling3d_1/split_1:output:36)u_net3d/up_sampling3d_1/split_1:output:37)u_net3d/up_sampling3d_1/split_1:output:37)u_net3d/up_sampling3d_1/split_1:output:38)u_net3d/up_sampling3d_1/split_1:output:38)u_net3d/up_sampling3d_1/split_1:output:39)u_net3d/up_sampling3d_1/split_1:output:39)u_net3d/up_sampling3d_1/split_1:output:40)u_net3d/up_sampling3d_1/split_1:output:40)u_net3d/up_sampling3d_1/split_1:output:41)u_net3d/up_sampling3d_1/split_1:output:41)u_net3d/up_sampling3d_1/split_1:output:42)u_net3d/up_sampling3d_1/split_1:output:42)u_net3d/up_sampling3d_1/split_1:output:43)u_net3d/up_sampling3d_1/split_1:output:43)u_net3d/up_sampling3d_1/split_1:output:44)u_net3d/up_sampling3d_1/split_1:output:44)u_net3d/up_sampling3d_1/split_1:output:45)u_net3d/up_sampling3d_1/split_1:output:45)u_net3d/up_sampling3d_1/split_1:output:46)u_net3d/up_sampling3d_1/split_1:output:46)u_net3d/up_sampling3d_1/split_1:output:47)u_net3d/up_sampling3d_1/split_1:output:47)u_net3d/up_sampling3d_1/split_1:output:48)u_net3d/up_sampling3d_1/split_1:output:48)u_net3d/up_sampling3d_1/split_1:output:49)u_net3d/up_sampling3d_1/split_1:output:49)u_net3d/up_sampling3d_1/split_1:output:50)u_net3d/up_sampling3d_1/split_1:output:50)u_net3d/up_sampling3d_1/split_1:output:51)u_net3d/up_sampling3d_1/split_1:output:51)u_net3d/up_sampling3d_1/split_1:output:52)u_net3d/up_sampling3d_1/split_1:output:52)u_net3d/up_sampling3d_1/split_1:output:53)u_net3d/up_sampling3d_1/split_1:output:53)u_net3d/up_sampling3d_1/split_1:output:54)u_net3d/up_sampling3d_1/split_1:output:54)u_net3d/up_sampling3d_1/split_1:output:55)u_net3d/up_sampling3d_1/split_1:output:55)u_net3d/up_sampling3d_1/split_1:output:56)u_net3d/up_sampling3d_1/split_1:output:56)u_net3d/up_sampling3d_1/split_1:output:57)u_net3d/up_sampling3d_1/split_1:output:57)u_net3d/up_sampling3d_1/split_1:output:58)u_net3d/up_sampling3d_1/split_1:output:58)u_net3d/up_sampling3d_1/split_1:output:59)u_net3d/up_sampling3d_1/split_1:output:59)u_net3d/up_sampling3d_1/split_1:output:60)u_net3d/up_sampling3d_1/split_1:output:60)u_net3d/up_sampling3d_1/split_1:output:61)u_net3d/up_sampling3d_1/split_1:output:61)u_net3d/up_sampling3d_1/split_1:output:62)u_net3d/up_sampling3d_1/split_1:output:62)u_net3d/up_sampling3d_1/split_1:output:63)u_net3d/up_sampling3d_1/split_1:output:63.u_net3d/up_sampling3d_1/concat_1/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’
 k
)u_net3d/up_sampling3d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ś	
u_net3d/up_sampling3d_1/split_2Split2u_net3d/up_sampling3d_1/split_2/split_dim:output:0)u_net3d/up_sampling3d_1/concat_1:output:0*
T0*¶
_output_shapes£
 :’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
*
	num_split g
%u_net3d/up_sampling3d_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :Ń
 u_net3d/up_sampling3d_1/concat_2ConcatV2(u_net3d/up_sampling3d_1/split_2:output:0(u_net3d/up_sampling3d_1/split_2:output:0(u_net3d/up_sampling3d_1/split_2:output:1(u_net3d/up_sampling3d_1/split_2:output:1(u_net3d/up_sampling3d_1/split_2:output:2(u_net3d/up_sampling3d_1/split_2:output:2(u_net3d/up_sampling3d_1/split_2:output:3(u_net3d/up_sampling3d_1/split_2:output:3(u_net3d/up_sampling3d_1/split_2:output:4(u_net3d/up_sampling3d_1/split_2:output:4(u_net3d/up_sampling3d_1/split_2:output:5(u_net3d/up_sampling3d_1/split_2:output:5(u_net3d/up_sampling3d_1/split_2:output:6(u_net3d/up_sampling3d_1/split_2:output:6(u_net3d/up_sampling3d_1/split_2:output:7(u_net3d/up_sampling3d_1/split_2:output:7(u_net3d/up_sampling3d_1/split_2:output:8(u_net3d/up_sampling3d_1/split_2:output:8(u_net3d/up_sampling3d_1/split_2:output:9(u_net3d/up_sampling3d_1/split_2:output:9)u_net3d/up_sampling3d_1/split_2:output:10)u_net3d/up_sampling3d_1/split_2:output:10)u_net3d/up_sampling3d_1/split_2:output:11)u_net3d/up_sampling3d_1/split_2:output:11)u_net3d/up_sampling3d_1/split_2:output:12)u_net3d/up_sampling3d_1/split_2:output:12)u_net3d/up_sampling3d_1/split_2:output:13)u_net3d/up_sampling3d_1/split_2:output:13)u_net3d/up_sampling3d_1/split_2:output:14)u_net3d/up_sampling3d_1/split_2:output:14)u_net3d/up_sampling3d_1/split_2:output:15)u_net3d/up_sampling3d_1/split_2:output:15)u_net3d/up_sampling3d_1/split_2:output:16)u_net3d/up_sampling3d_1/split_2:output:16)u_net3d/up_sampling3d_1/split_2:output:17)u_net3d/up_sampling3d_1/split_2:output:17)u_net3d/up_sampling3d_1/split_2:output:18)u_net3d/up_sampling3d_1/split_2:output:18)u_net3d/up_sampling3d_1/split_2:output:19)u_net3d/up_sampling3d_1/split_2:output:19)u_net3d/up_sampling3d_1/split_2:output:20)u_net3d/up_sampling3d_1/split_2:output:20)u_net3d/up_sampling3d_1/split_2:output:21)u_net3d/up_sampling3d_1/split_2:output:21)u_net3d/up_sampling3d_1/split_2:output:22)u_net3d/up_sampling3d_1/split_2:output:22)u_net3d/up_sampling3d_1/split_2:output:23)u_net3d/up_sampling3d_1/split_2:output:23)u_net3d/up_sampling3d_1/split_2:output:24)u_net3d/up_sampling3d_1/split_2:output:24)u_net3d/up_sampling3d_1/split_2:output:25)u_net3d/up_sampling3d_1/split_2:output:25)u_net3d/up_sampling3d_1/split_2:output:26)u_net3d/up_sampling3d_1/split_2:output:26)u_net3d/up_sampling3d_1/split_2:output:27)u_net3d/up_sampling3d_1/split_2:output:27)u_net3d/up_sampling3d_1/split_2:output:28)u_net3d/up_sampling3d_1/split_2:output:28)u_net3d/up_sampling3d_1/split_2:output:29)u_net3d/up_sampling3d_1/split_2:output:29)u_net3d/up_sampling3d_1/split_2:output:30)u_net3d/up_sampling3d_1/split_2:output:30)u_net3d/up_sampling3d_1/split_2:output:31)u_net3d/up_sampling3d_1/split_2:output:31.u_net3d/up_sampling3d_1/concat_2/axis:output:0*
N@*
T0*5
_output_shapes#
!:’’’’’’’’’
@c
!u_net3d/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :õ
u_net3d/concatenate_1/concatConcatV2)u_net3d/up_sampling3d_1/concat_2:output:0+u_net3d/conv_block3d_2/Relu_1:activations:0*u_net3d/concatenate_1/concat/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’
@ĄĮ
5u_net3d/conv_block3d_3/conv3d_6/Conv3D/ReadVariableOpReadVariableOp>u_net3d_conv_block3d_3_conv3d_6_conv3d_readvariableop_resource*+
_output_shapes
:Ą@*
dtype0ž
&u_net3d/conv_block3d_3/conv3d_6/Conv3DConv3D%u_net3d/concatenate_1/concat:output:0=u_net3d/conv_block3d_3/conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
²
6u_net3d/conv_block3d_3/conv3d_6/BiasAdd/ReadVariableOpReadVariableOp?u_net3d_conv_block3d_3_conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ā
'u_net3d/conv_block3d_3/conv3d_6/BiasAddBiasAdd/u_net3d/conv_block3d_3/conv3d_6/Conv3D:output:0>u_net3d/conv_block3d_3/conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
u_net3d/conv_block3d_3/ReluRelu0u_net3d/conv_block3d_3/conv3d_6/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@Ą
5u_net3d/conv_block3d_3/conv3d_7/Conv3D/ReadVariableOpReadVariableOp>u_net3d_conv_block3d_3_conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0
&u_net3d/conv_block3d_3/conv3d_7/Conv3DConv3D)u_net3d/conv_block3d_3/Relu:activations:0=u_net3d/conv_block3d_3/conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
²
6u_net3d/conv_block3d_3/conv3d_7/BiasAdd/ReadVariableOpReadVariableOp?u_net3d_conv_block3d_3_conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ā
'u_net3d/conv_block3d_3/conv3d_7/BiasAddBiasAdd/u_net3d/conv_block3d_3/conv3d_7/Conv3D:output:0>u_net3d/conv_block3d_3/conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
u_net3d/conv_block3d_3/Relu_1Relu0u_net3d/conv_block3d_3/conv3d_7/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@g
%u_net3d/up_sampling3d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ō
u_net3d/up_sampling3d/splitSplit.u_net3d/up_sampling3d/split/split_dim:output:0+u_net3d/conv_block3d_3/Relu_1:activations:0*
T0*Ö
_output_shapesĆ
Ą:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@*
	num_split
c
!u_net3d/up_sampling3d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
u_net3d/up_sampling3d/concatConcatV2$u_net3d/up_sampling3d/split:output:0$u_net3d/up_sampling3d/split:output:0$u_net3d/up_sampling3d/split:output:1$u_net3d/up_sampling3d/split:output:1$u_net3d/up_sampling3d/split:output:2$u_net3d/up_sampling3d/split:output:2$u_net3d/up_sampling3d/split:output:3$u_net3d/up_sampling3d/split:output:3$u_net3d/up_sampling3d/split:output:4$u_net3d/up_sampling3d/split:output:4$u_net3d/up_sampling3d/split:output:5$u_net3d/up_sampling3d/split:output:5$u_net3d/up_sampling3d/split:output:6$u_net3d/up_sampling3d/split:output:6$u_net3d/up_sampling3d/split:output:7$u_net3d/up_sampling3d/split:output:7$u_net3d/up_sampling3d/split:output:8$u_net3d/up_sampling3d/split:output:8$u_net3d/up_sampling3d/split:output:9$u_net3d/up_sampling3d/split:output:9*u_net3d/up_sampling3d/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :’’’’’’’’’@@i
'u_net3d/up_sampling3d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :³ 
u_net3d/up_sampling3d/split_1Split0u_net3d/up_sampling3d/split_1/split_dim:output:0%u_net3d/up_sampling3d/concat:output:0*
T0*
_output_shapes
:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@*
	num_splite
#u_net3d/up_sampling3d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :ÅS
u_net3d/up_sampling3d/concat_1ConcatV2&u_net3d/up_sampling3d/split_1:output:0&u_net3d/up_sampling3d/split_1:output:0&u_net3d/up_sampling3d/split_1:output:1&u_net3d/up_sampling3d/split_1:output:1&u_net3d/up_sampling3d/split_1:output:2&u_net3d/up_sampling3d/split_1:output:2&u_net3d/up_sampling3d/split_1:output:3&u_net3d/up_sampling3d/split_1:output:3&u_net3d/up_sampling3d/split_1:output:4&u_net3d/up_sampling3d/split_1:output:4&u_net3d/up_sampling3d/split_1:output:5&u_net3d/up_sampling3d/split_1:output:5&u_net3d/up_sampling3d/split_1:output:6&u_net3d/up_sampling3d/split_1:output:6&u_net3d/up_sampling3d/split_1:output:7&u_net3d/up_sampling3d/split_1:output:7&u_net3d/up_sampling3d/split_1:output:8&u_net3d/up_sampling3d/split_1:output:8&u_net3d/up_sampling3d/split_1:output:9&u_net3d/up_sampling3d/split_1:output:9'u_net3d/up_sampling3d/split_1:output:10'u_net3d/up_sampling3d/split_1:output:10'u_net3d/up_sampling3d/split_1:output:11'u_net3d/up_sampling3d/split_1:output:11'u_net3d/up_sampling3d/split_1:output:12'u_net3d/up_sampling3d/split_1:output:12'u_net3d/up_sampling3d/split_1:output:13'u_net3d/up_sampling3d/split_1:output:13'u_net3d/up_sampling3d/split_1:output:14'u_net3d/up_sampling3d/split_1:output:14'u_net3d/up_sampling3d/split_1:output:15'u_net3d/up_sampling3d/split_1:output:15'u_net3d/up_sampling3d/split_1:output:16'u_net3d/up_sampling3d/split_1:output:16'u_net3d/up_sampling3d/split_1:output:17'u_net3d/up_sampling3d/split_1:output:17'u_net3d/up_sampling3d/split_1:output:18'u_net3d/up_sampling3d/split_1:output:18'u_net3d/up_sampling3d/split_1:output:19'u_net3d/up_sampling3d/split_1:output:19'u_net3d/up_sampling3d/split_1:output:20'u_net3d/up_sampling3d/split_1:output:20'u_net3d/up_sampling3d/split_1:output:21'u_net3d/up_sampling3d/split_1:output:21'u_net3d/up_sampling3d/split_1:output:22'u_net3d/up_sampling3d/split_1:output:22'u_net3d/up_sampling3d/split_1:output:23'u_net3d/up_sampling3d/split_1:output:23'u_net3d/up_sampling3d/split_1:output:24'u_net3d/up_sampling3d/split_1:output:24'u_net3d/up_sampling3d/split_1:output:25'u_net3d/up_sampling3d/split_1:output:25'u_net3d/up_sampling3d/split_1:output:26'u_net3d/up_sampling3d/split_1:output:26'u_net3d/up_sampling3d/split_1:output:27'u_net3d/up_sampling3d/split_1:output:27'u_net3d/up_sampling3d/split_1:output:28'u_net3d/up_sampling3d/split_1:output:28'u_net3d/up_sampling3d/split_1:output:29'u_net3d/up_sampling3d/split_1:output:29'u_net3d/up_sampling3d/split_1:output:30'u_net3d/up_sampling3d/split_1:output:30'u_net3d/up_sampling3d/split_1:output:31'u_net3d/up_sampling3d/split_1:output:31'u_net3d/up_sampling3d/split_1:output:32'u_net3d/up_sampling3d/split_1:output:32'u_net3d/up_sampling3d/split_1:output:33'u_net3d/up_sampling3d/split_1:output:33'u_net3d/up_sampling3d/split_1:output:34'u_net3d/up_sampling3d/split_1:output:34'u_net3d/up_sampling3d/split_1:output:35'u_net3d/up_sampling3d/split_1:output:35'u_net3d/up_sampling3d/split_1:output:36'u_net3d/up_sampling3d/split_1:output:36'u_net3d/up_sampling3d/split_1:output:37'u_net3d/up_sampling3d/split_1:output:37'u_net3d/up_sampling3d/split_1:output:38'u_net3d/up_sampling3d/split_1:output:38'u_net3d/up_sampling3d/split_1:output:39'u_net3d/up_sampling3d/split_1:output:39'u_net3d/up_sampling3d/split_1:output:40'u_net3d/up_sampling3d/split_1:output:40'u_net3d/up_sampling3d/split_1:output:41'u_net3d/up_sampling3d/split_1:output:41'u_net3d/up_sampling3d/split_1:output:42'u_net3d/up_sampling3d/split_1:output:42'u_net3d/up_sampling3d/split_1:output:43'u_net3d/up_sampling3d/split_1:output:43'u_net3d/up_sampling3d/split_1:output:44'u_net3d/up_sampling3d/split_1:output:44'u_net3d/up_sampling3d/split_1:output:45'u_net3d/up_sampling3d/split_1:output:45'u_net3d/up_sampling3d/split_1:output:46'u_net3d/up_sampling3d/split_1:output:46'u_net3d/up_sampling3d/split_1:output:47'u_net3d/up_sampling3d/split_1:output:47'u_net3d/up_sampling3d/split_1:output:48'u_net3d/up_sampling3d/split_1:output:48'u_net3d/up_sampling3d/split_1:output:49'u_net3d/up_sampling3d/split_1:output:49'u_net3d/up_sampling3d/split_1:output:50'u_net3d/up_sampling3d/split_1:output:50'u_net3d/up_sampling3d/split_1:output:51'u_net3d/up_sampling3d/split_1:output:51'u_net3d/up_sampling3d/split_1:output:52'u_net3d/up_sampling3d/split_1:output:52'u_net3d/up_sampling3d/split_1:output:53'u_net3d/up_sampling3d/split_1:output:53'u_net3d/up_sampling3d/split_1:output:54'u_net3d/up_sampling3d/split_1:output:54'u_net3d/up_sampling3d/split_1:output:55'u_net3d/up_sampling3d/split_1:output:55'u_net3d/up_sampling3d/split_1:output:56'u_net3d/up_sampling3d/split_1:output:56'u_net3d/up_sampling3d/split_1:output:57'u_net3d/up_sampling3d/split_1:output:57'u_net3d/up_sampling3d/split_1:output:58'u_net3d/up_sampling3d/split_1:output:58'u_net3d/up_sampling3d/split_1:output:59'u_net3d/up_sampling3d/split_1:output:59'u_net3d/up_sampling3d/split_1:output:60'u_net3d/up_sampling3d/split_1:output:60'u_net3d/up_sampling3d/split_1:output:61'u_net3d/up_sampling3d/split_1:output:61'u_net3d/up_sampling3d/split_1:output:62'u_net3d/up_sampling3d/split_1:output:62'u_net3d/up_sampling3d/split_1:output:63'u_net3d/up_sampling3d/split_1:output:63'u_net3d/up_sampling3d/split_1:output:64'u_net3d/up_sampling3d/split_1:output:64'u_net3d/up_sampling3d/split_1:output:65'u_net3d/up_sampling3d/split_1:output:65'u_net3d/up_sampling3d/split_1:output:66'u_net3d/up_sampling3d/split_1:output:66'u_net3d/up_sampling3d/split_1:output:67'u_net3d/up_sampling3d/split_1:output:67'u_net3d/up_sampling3d/split_1:output:68'u_net3d/up_sampling3d/split_1:output:68'u_net3d/up_sampling3d/split_1:output:69'u_net3d/up_sampling3d/split_1:output:69'u_net3d/up_sampling3d/split_1:output:70'u_net3d/up_sampling3d/split_1:output:70'u_net3d/up_sampling3d/split_1:output:71'u_net3d/up_sampling3d/split_1:output:71'u_net3d/up_sampling3d/split_1:output:72'u_net3d/up_sampling3d/split_1:output:72'u_net3d/up_sampling3d/split_1:output:73'u_net3d/up_sampling3d/split_1:output:73'u_net3d/up_sampling3d/split_1:output:74'u_net3d/up_sampling3d/split_1:output:74'u_net3d/up_sampling3d/split_1:output:75'u_net3d/up_sampling3d/split_1:output:75'u_net3d/up_sampling3d/split_1:output:76'u_net3d/up_sampling3d/split_1:output:76'u_net3d/up_sampling3d/split_1:output:77'u_net3d/up_sampling3d/split_1:output:77'u_net3d/up_sampling3d/split_1:output:78'u_net3d/up_sampling3d/split_1:output:78'u_net3d/up_sampling3d/split_1:output:79'u_net3d/up_sampling3d/split_1:output:79'u_net3d/up_sampling3d/split_1:output:80'u_net3d/up_sampling3d/split_1:output:80'u_net3d/up_sampling3d/split_1:output:81'u_net3d/up_sampling3d/split_1:output:81'u_net3d/up_sampling3d/split_1:output:82'u_net3d/up_sampling3d/split_1:output:82'u_net3d/up_sampling3d/split_1:output:83'u_net3d/up_sampling3d/split_1:output:83'u_net3d/up_sampling3d/split_1:output:84'u_net3d/up_sampling3d/split_1:output:84'u_net3d/up_sampling3d/split_1:output:85'u_net3d/up_sampling3d/split_1:output:85'u_net3d/up_sampling3d/split_1:output:86'u_net3d/up_sampling3d/split_1:output:86'u_net3d/up_sampling3d/split_1:output:87'u_net3d/up_sampling3d/split_1:output:87'u_net3d/up_sampling3d/split_1:output:88'u_net3d/up_sampling3d/split_1:output:88'u_net3d/up_sampling3d/split_1:output:89'u_net3d/up_sampling3d/split_1:output:89'u_net3d/up_sampling3d/split_1:output:90'u_net3d/up_sampling3d/split_1:output:90'u_net3d/up_sampling3d/split_1:output:91'u_net3d/up_sampling3d/split_1:output:91'u_net3d/up_sampling3d/split_1:output:92'u_net3d/up_sampling3d/split_1:output:92'u_net3d/up_sampling3d/split_1:output:93'u_net3d/up_sampling3d/split_1:output:93'u_net3d/up_sampling3d/split_1:output:94'u_net3d/up_sampling3d/split_1:output:94'u_net3d/up_sampling3d/split_1:output:95'u_net3d/up_sampling3d/split_1:output:95'u_net3d/up_sampling3d/split_1:output:96'u_net3d/up_sampling3d/split_1:output:96'u_net3d/up_sampling3d/split_1:output:97'u_net3d/up_sampling3d/split_1:output:97'u_net3d/up_sampling3d/split_1:output:98'u_net3d/up_sampling3d/split_1:output:98'u_net3d/up_sampling3d/split_1:output:99'u_net3d/up_sampling3d/split_1:output:99(u_net3d/up_sampling3d/split_1:output:100(u_net3d/up_sampling3d/split_1:output:100(u_net3d/up_sampling3d/split_1:output:101(u_net3d/up_sampling3d/split_1:output:101(u_net3d/up_sampling3d/split_1:output:102(u_net3d/up_sampling3d/split_1:output:102(u_net3d/up_sampling3d/split_1:output:103(u_net3d/up_sampling3d/split_1:output:103(u_net3d/up_sampling3d/split_1:output:104(u_net3d/up_sampling3d/split_1:output:104(u_net3d/up_sampling3d/split_1:output:105(u_net3d/up_sampling3d/split_1:output:105(u_net3d/up_sampling3d/split_1:output:106(u_net3d/up_sampling3d/split_1:output:106(u_net3d/up_sampling3d/split_1:output:107(u_net3d/up_sampling3d/split_1:output:107(u_net3d/up_sampling3d/split_1:output:108(u_net3d/up_sampling3d/split_1:output:108(u_net3d/up_sampling3d/split_1:output:109(u_net3d/up_sampling3d/split_1:output:109(u_net3d/up_sampling3d/split_1:output:110(u_net3d/up_sampling3d/split_1:output:110(u_net3d/up_sampling3d/split_1:output:111(u_net3d/up_sampling3d/split_1:output:111(u_net3d/up_sampling3d/split_1:output:112(u_net3d/up_sampling3d/split_1:output:112(u_net3d/up_sampling3d/split_1:output:113(u_net3d/up_sampling3d/split_1:output:113(u_net3d/up_sampling3d/split_1:output:114(u_net3d/up_sampling3d/split_1:output:114(u_net3d/up_sampling3d/split_1:output:115(u_net3d/up_sampling3d/split_1:output:115(u_net3d/up_sampling3d/split_1:output:116(u_net3d/up_sampling3d/split_1:output:116(u_net3d/up_sampling3d/split_1:output:117(u_net3d/up_sampling3d/split_1:output:117(u_net3d/up_sampling3d/split_1:output:118(u_net3d/up_sampling3d/split_1:output:118(u_net3d/up_sampling3d/split_1:output:119(u_net3d/up_sampling3d/split_1:output:119(u_net3d/up_sampling3d/split_1:output:120(u_net3d/up_sampling3d/split_1:output:120(u_net3d/up_sampling3d/split_1:output:121(u_net3d/up_sampling3d/split_1:output:121(u_net3d/up_sampling3d/split_1:output:122(u_net3d/up_sampling3d/split_1:output:122(u_net3d/up_sampling3d/split_1:output:123(u_net3d/up_sampling3d/split_1:output:123(u_net3d/up_sampling3d/split_1:output:124(u_net3d/up_sampling3d/split_1:output:124(u_net3d/up_sampling3d/split_1:output:125(u_net3d/up_sampling3d/split_1:output:125(u_net3d/up_sampling3d/split_1:output:126(u_net3d/up_sampling3d/split_1:output:126(u_net3d/up_sampling3d/split_1:output:127(u_net3d/up_sampling3d/split_1:output:127,u_net3d/up_sampling3d/concat_1/axis:output:0*
N*
T0*4
_output_shapes"
 :’’’’’’’’’@@i
'u_net3d/up_sampling3d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :“
u_net3d/up_sampling3d/split_2Split0u_net3d/up_sampling3d/split_2/split_dim:output:0'u_net3d/up_sampling3d/concat_1:output:0*
T0*
_output_shapes
:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@*
	num_split@e
#u_net3d/up_sampling3d/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :*
u_net3d/up_sampling3d/concat_2ConcatV2&u_net3d/up_sampling3d/split_2:output:0&u_net3d/up_sampling3d/split_2:output:0&u_net3d/up_sampling3d/split_2:output:1&u_net3d/up_sampling3d/split_2:output:1&u_net3d/up_sampling3d/split_2:output:2&u_net3d/up_sampling3d/split_2:output:2&u_net3d/up_sampling3d/split_2:output:3&u_net3d/up_sampling3d/split_2:output:3&u_net3d/up_sampling3d/split_2:output:4&u_net3d/up_sampling3d/split_2:output:4&u_net3d/up_sampling3d/split_2:output:5&u_net3d/up_sampling3d/split_2:output:5&u_net3d/up_sampling3d/split_2:output:6&u_net3d/up_sampling3d/split_2:output:6&u_net3d/up_sampling3d/split_2:output:7&u_net3d/up_sampling3d/split_2:output:7&u_net3d/up_sampling3d/split_2:output:8&u_net3d/up_sampling3d/split_2:output:8&u_net3d/up_sampling3d/split_2:output:9&u_net3d/up_sampling3d/split_2:output:9'u_net3d/up_sampling3d/split_2:output:10'u_net3d/up_sampling3d/split_2:output:10'u_net3d/up_sampling3d/split_2:output:11'u_net3d/up_sampling3d/split_2:output:11'u_net3d/up_sampling3d/split_2:output:12'u_net3d/up_sampling3d/split_2:output:12'u_net3d/up_sampling3d/split_2:output:13'u_net3d/up_sampling3d/split_2:output:13'u_net3d/up_sampling3d/split_2:output:14'u_net3d/up_sampling3d/split_2:output:14'u_net3d/up_sampling3d/split_2:output:15'u_net3d/up_sampling3d/split_2:output:15'u_net3d/up_sampling3d/split_2:output:16'u_net3d/up_sampling3d/split_2:output:16'u_net3d/up_sampling3d/split_2:output:17'u_net3d/up_sampling3d/split_2:output:17'u_net3d/up_sampling3d/split_2:output:18'u_net3d/up_sampling3d/split_2:output:18'u_net3d/up_sampling3d/split_2:output:19'u_net3d/up_sampling3d/split_2:output:19'u_net3d/up_sampling3d/split_2:output:20'u_net3d/up_sampling3d/split_2:output:20'u_net3d/up_sampling3d/split_2:output:21'u_net3d/up_sampling3d/split_2:output:21'u_net3d/up_sampling3d/split_2:output:22'u_net3d/up_sampling3d/split_2:output:22'u_net3d/up_sampling3d/split_2:output:23'u_net3d/up_sampling3d/split_2:output:23'u_net3d/up_sampling3d/split_2:output:24'u_net3d/up_sampling3d/split_2:output:24'u_net3d/up_sampling3d/split_2:output:25'u_net3d/up_sampling3d/split_2:output:25'u_net3d/up_sampling3d/split_2:output:26'u_net3d/up_sampling3d/split_2:output:26'u_net3d/up_sampling3d/split_2:output:27'u_net3d/up_sampling3d/split_2:output:27'u_net3d/up_sampling3d/split_2:output:28'u_net3d/up_sampling3d/split_2:output:28'u_net3d/up_sampling3d/split_2:output:29'u_net3d/up_sampling3d/split_2:output:29'u_net3d/up_sampling3d/split_2:output:30'u_net3d/up_sampling3d/split_2:output:30'u_net3d/up_sampling3d/split_2:output:31'u_net3d/up_sampling3d/split_2:output:31'u_net3d/up_sampling3d/split_2:output:32'u_net3d/up_sampling3d/split_2:output:32'u_net3d/up_sampling3d/split_2:output:33'u_net3d/up_sampling3d/split_2:output:33'u_net3d/up_sampling3d/split_2:output:34'u_net3d/up_sampling3d/split_2:output:34'u_net3d/up_sampling3d/split_2:output:35'u_net3d/up_sampling3d/split_2:output:35'u_net3d/up_sampling3d/split_2:output:36'u_net3d/up_sampling3d/split_2:output:36'u_net3d/up_sampling3d/split_2:output:37'u_net3d/up_sampling3d/split_2:output:37'u_net3d/up_sampling3d/split_2:output:38'u_net3d/up_sampling3d/split_2:output:38'u_net3d/up_sampling3d/split_2:output:39'u_net3d/up_sampling3d/split_2:output:39'u_net3d/up_sampling3d/split_2:output:40'u_net3d/up_sampling3d/split_2:output:40'u_net3d/up_sampling3d/split_2:output:41'u_net3d/up_sampling3d/split_2:output:41'u_net3d/up_sampling3d/split_2:output:42'u_net3d/up_sampling3d/split_2:output:42'u_net3d/up_sampling3d/split_2:output:43'u_net3d/up_sampling3d/split_2:output:43'u_net3d/up_sampling3d/split_2:output:44'u_net3d/up_sampling3d/split_2:output:44'u_net3d/up_sampling3d/split_2:output:45'u_net3d/up_sampling3d/split_2:output:45'u_net3d/up_sampling3d/split_2:output:46'u_net3d/up_sampling3d/split_2:output:46'u_net3d/up_sampling3d/split_2:output:47'u_net3d/up_sampling3d/split_2:output:47'u_net3d/up_sampling3d/split_2:output:48'u_net3d/up_sampling3d/split_2:output:48'u_net3d/up_sampling3d/split_2:output:49'u_net3d/up_sampling3d/split_2:output:49'u_net3d/up_sampling3d/split_2:output:50'u_net3d/up_sampling3d/split_2:output:50'u_net3d/up_sampling3d/split_2:output:51'u_net3d/up_sampling3d/split_2:output:51'u_net3d/up_sampling3d/split_2:output:52'u_net3d/up_sampling3d/split_2:output:52'u_net3d/up_sampling3d/split_2:output:53'u_net3d/up_sampling3d/split_2:output:53'u_net3d/up_sampling3d/split_2:output:54'u_net3d/up_sampling3d/split_2:output:54'u_net3d/up_sampling3d/split_2:output:55'u_net3d/up_sampling3d/split_2:output:55'u_net3d/up_sampling3d/split_2:output:56'u_net3d/up_sampling3d/split_2:output:56'u_net3d/up_sampling3d/split_2:output:57'u_net3d/up_sampling3d/split_2:output:57'u_net3d/up_sampling3d/split_2:output:58'u_net3d/up_sampling3d/split_2:output:58'u_net3d/up_sampling3d/split_2:output:59'u_net3d/up_sampling3d/split_2:output:59'u_net3d/up_sampling3d/split_2:output:60'u_net3d/up_sampling3d/split_2:output:60'u_net3d/up_sampling3d/split_2:output:61'u_net3d/up_sampling3d/split_2:output:61'u_net3d/up_sampling3d/split_2:output:62'u_net3d/up_sampling3d/split_2:output:62'u_net3d/up_sampling3d/split_2:output:63'u_net3d/up_sampling3d/split_2:output:63,u_net3d/up_sampling3d/concat_2/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’@a
u_net3d/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ķ
u_net3d/concatenate/concatConcatV2'u_net3d/up_sampling3d/concat_2:output:0)u_net3d/conv_block3d/Relu_1:activations:0(u_net3d/concatenate/concat/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’`Ą
5u_net3d/conv_block3d_1/conv3d_2/Conv3D/ReadVariableOpReadVariableOp>u_net3d_conv_block3d_1_conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:` *
dtype0ż
&u_net3d/conv_block3d_1/conv3d_2/Conv3DConv3D#u_net3d/concatenate/concat:output:0=u_net3d/conv_block3d_1/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	
²
6u_net3d/conv_block3d_1/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp?u_net3d_conv_block3d_1_conv3d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ć
'u_net3d/conv_block3d_1/conv3d_2/BiasAddBiasAdd/u_net3d/conv_block3d_1/conv3d_2/Conv3D:output:0>u_net3d/conv_block3d_1/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
u_net3d/conv_block3d_1/ReluRelu0u_net3d/conv_block3d_1/conv3d_2/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ Ą
5u_net3d/conv_block3d_1/conv3d_3/Conv3D/ReadVariableOpReadVariableOp>u_net3d_conv_block3d_1_conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0
&u_net3d/conv_block3d_1/conv3d_3/Conv3DConv3D)u_net3d/conv_block3d_1/Relu:activations:0=u_net3d/conv_block3d_1/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	
²
6u_net3d/conv_block3d_1/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp?u_net3d_conv_block3d_1_conv3d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ć
'u_net3d/conv_block3d_1/conv3d_3/BiasAddBiasAdd/u_net3d/conv_block3d_1/conv3d_3/Conv3D:output:0>u_net3d/conv_block3d_1/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
u_net3d/conv_block3d_1/Relu_1Relu0u_net3d/conv_block3d_1/conv3d_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ Ā
6u_net3d/conv_block3d_5/conv3d_10/Conv3D/ReadVariableOpReadVariableOp?u_net3d_conv_block3d_5_conv3d_10_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0
'u_net3d/conv_block3d_5/conv3d_10/Conv3DConv3D+u_net3d/conv_block3d_1/Relu_1:activations:0>u_net3d/conv_block3d_5/conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’*
paddingSAME*
strides	
“
7u_net3d/conv_block3d_5/conv3d_10/BiasAdd/ReadVariableOpReadVariableOp@u_net3d_conv_block3d_5_conv3d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ę
(u_net3d/conv_block3d_5/conv3d_10/BiasAddBiasAdd0u_net3d/conv_block3d_5/conv3d_10/Conv3D:output:0?u_net3d/conv_block3d_5/conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’
IdentityIdentity1u_net3d/conv_block3d_5/conv3d_10/BiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’

NoOpNoOp3^u_net3d/conv_block3d/conv3d/BiasAdd/ReadVariableOp2^u_net3d/conv_block3d/conv3d/Conv3D/ReadVariableOp5^u_net3d/conv_block3d/conv3d_1/BiasAdd/ReadVariableOp4^u_net3d/conv_block3d/conv3d_1/Conv3D/ReadVariableOp7^u_net3d/conv_block3d_1/conv3d_2/BiasAdd/ReadVariableOp6^u_net3d/conv_block3d_1/conv3d_2/Conv3D/ReadVariableOp7^u_net3d/conv_block3d_1/conv3d_3/BiasAdd/ReadVariableOp6^u_net3d/conv_block3d_1/conv3d_3/Conv3D/ReadVariableOp7^u_net3d/conv_block3d_2/conv3d_4/BiasAdd/ReadVariableOp6^u_net3d/conv_block3d_2/conv3d_4/Conv3D/ReadVariableOp7^u_net3d/conv_block3d_2/conv3d_5/BiasAdd/ReadVariableOp6^u_net3d/conv_block3d_2/conv3d_5/Conv3D/ReadVariableOp7^u_net3d/conv_block3d_3/conv3d_6/BiasAdd/ReadVariableOp6^u_net3d/conv_block3d_3/conv3d_6/Conv3D/ReadVariableOp7^u_net3d/conv_block3d_3/conv3d_7/BiasAdd/ReadVariableOp6^u_net3d/conv_block3d_3/conv3d_7/Conv3D/ReadVariableOp7^u_net3d/conv_block3d_4/conv3d_8/BiasAdd/ReadVariableOp6^u_net3d/conv_block3d_4/conv3d_8/Conv3D/ReadVariableOp7^u_net3d/conv_block3d_4/conv3d_9/BiasAdd/ReadVariableOp6^u_net3d/conv_block3d_4/conv3d_9/Conv3D/ReadVariableOp8^u_net3d/conv_block3d_5/conv3d_10/BiasAdd/ReadVariableOp7^u_net3d/conv_block3d_5/conv3d_10/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : 2h
2u_net3d/conv_block3d/conv3d/BiasAdd/ReadVariableOp2u_net3d/conv_block3d/conv3d/BiasAdd/ReadVariableOp2f
1u_net3d/conv_block3d/conv3d/Conv3D/ReadVariableOp1u_net3d/conv_block3d/conv3d/Conv3D/ReadVariableOp2l
4u_net3d/conv_block3d/conv3d_1/BiasAdd/ReadVariableOp4u_net3d/conv_block3d/conv3d_1/BiasAdd/ReadVariableOp2j
3u_net3d/conv_block3d/conv3d_1/Conv3D/ReadVariableOp3u_net3d/conv_block3d/conv3d_1/Conv3D/ReadVariableOp2p
6u_net3d/conv_block3d_1/conv3d_2/BiasAdd/ReadVariableOp6u_net3d/conv_block3d_1/conv3d_2/BiasAdd/ReadVariableOp2n
5u_net3d/conv_block3d_1/conv3d_2/Conv3D/ReadVariableOp5u_net3d/conv_block3d_1/conv3d_2/Conv3D/ReadVariableOp2p
6u_net3d/conv_block3d_1/conv3d_3/BiasAdd/ReadVariableOp6u_net3d/conv_block3d_1/conv3d_3/BiasAdd/ReadVariableOp2n
5u_net3d/conv_block3d_1/conv3d_3/Conv3D/ReadVariableOp5u_net3d/conv_block3d_1/conv3d_3/Conv3D/ReadVariableOp2p
6u_net3d/conv_block3d_2/conv3d_4/BiasAdd/ReadVariableOp6u_net3d/conv_block3d_2/conv3d_4/BiasAdd/ReadVariableOp2n
5u_net3d/conv_block3d_2/conv3d_4/Conv3D/ReadVariableOp5u_net3d/conv_block3d_2/conv3d_4/Conv3D/ReadVariableOp2p
6u_net3d/conv_block3d_2/conv3d_5/BiasAdd/ReadVariableOp6u_net3d/conv_block3d_2/conv3d_5/BiasAdd/ReadVariableOp2n
5u_net3d/conv_block3d_2/conv3d_5/Conv3D/ReadVariableOp5u_net3d/conv_block3d_2/conv3d_5/Conv3D/ReadVariableOp2p
6u_net3d/conv_block3d_3/conv3d_6/BiasAdd/ReadVariableOp6u_net3d/conv_block3d_3/conv3d_6/BiasAdd/ReadVariableOp2n
5u_net3d/conv_block3d_3/conv3d_6/Conv3D/ReadVariableOp5u_net3d/conv_block3d_3/conv3d_6/Conv3D/ReadVariableOp2p
6u_net3d/conv_block3d_3/conv3d_7/BiasAdd/ReadVariableOp6u_net3d/conv_block3d_3/conv3d_7/BiasAdd/ReadVariableOp2n
5u_net3d/conv_block3d_3/conv3d_7/Conv3D/ReadVariableOp5u_net3d/conv_block3d_3/conv3d_7/Conv3D/ReadVariableOp2p
6u_net3d/conv_block3d_4/conv3d_8/BiasAdd/ReadVariableOp6u_net3d/conv_block3d_4/conv3d_8/BiasAdd/ReadVariableOp2n
5u_net3d/conv_block3d_4/conv3d_8/Conv3D/ReadVariableOp5u_net3d/conv_block3d_4/conv3d_8/Conv3D/ReadVariableOp2p
6u_net3d/conv_block3d_4/conv3d_9/BiasAdd/ReadVariableOp6u_net3d/conv_block3d_4/conv3d_9/BiasAdd/ReadVariableOp2n
5u_net3d/conv_block3d_4/conv3d_9/Conv3D/ReadVariableOp5u_net3d/conv_block3d_4/conv3d_9/Conv3D/ReadVariableOp2r
7u_net3d/conv_block3d_5/conv3d_10/BiasAdd/ReadVariableOp7u_net3d/conv_block3d_5/conv3d_10/BiasAdd/ReadVariableOp2p
6u_net3d/conv_block3d_5/conv3d_10/Conv3D/ReadVariableOp6u_net3d/conv_block3d_5/conv3d_10/Conv3D/ReadVariableOp:^ Z
5
_output_shapes#
!:’’’’’’’’’
!
_user_specified_name	input_1
ų
ģ
0__inference_conv_block3d_2_layer_call_fn_1041250
input_1%
unknown: @
	unknown_0:@'
	unknown_1:@@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1041226|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:’’’’’’’’’
@ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :’’’’’’’’’
@ 
!
_user_specified_name	input_1
ĒC
h
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_1042031

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0inputs*
T0*¶
_output_shapes£
 :’’’’’’’’’@ :’’’’’’’’’@ :’’’’’’’’’@ :’’’’’’’’’@ :’’’’’’’’’@ *
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4concat/axis:output:0*
N
*
T0*4
_output_shapes"
 :’’’’’’’’’
@ S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :š
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*
_output_shapes
:’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 :’’’’’’’’’
 *
	num_split@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :ā
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15split_1:output:16split_1:output:16split_1:output:17split_1:output:17split_1:output:18split_1:output:18split_1:output:19split_1:output:19split_1:output:20split_1:output:20split_1:output:21split_1:output:21split_1:output:22split_1:output:22split_1:output:23split_1:output:23split_1:output:24split_1:output:24split_1:output:25split_1:output:25split_1:output:26split_1:output:26split_1:output:27split_1:output:27split_1:output:28split_1:output:28split_1:output:29split_1:output:29split_1:output:30split_1:output:30split_1:output:31split_1:output:31split_1:output:32split_1:output:32split_1:output:33split_1:output:33split_1:output:34split_1:output:34split_1:output:35split_1:output:35split_1:output:36split_1:output:36split_1:output:37split_1:output:37split_1:output:38split_1:output:38split_1:output:39split_1:output:39split_1:output:40split_1:output:40split_1:output:41split_1:output:41split_1:output:42split_1:output:42split_1:output:43split_1:output:43split_1:output:44split_1:output:44split_1:output:45split_1:output:45split_1:output:46split_1:output:46split_1:output:47split_1:output:47split_1:output:48split_1:output:48split_1:output:49split_1:output:49split_1:output:50split_1:output:50split_1:output:51split_1:output:51split_1:output:52split_1:output:52split_1:output:53split_1:output:53split_1:output:54split_1:output:54split_1:output:55split_1:output:55split_1:output:56split_1:output:56split_1:output:57split_1:output:57split_1:output:58split_1:output:58split_1:output:59split_1:output:59split_1:output:60split_1:output:60split_1:output:61split_1:output:61split_1:output:62split_1:output:62split_1:output:63split_1:output:63concat_1/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’
 S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :	
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*¶
_output_shapes£
 :’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
:’’’’’’’’’
*
	num_split O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :”

concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15split_2:output:16split_2:output:16split_2:output:17split_2:output:17split_2:output:18split_2:output:18split_2:output:19split_2:output:19split_2:output:20split_2:output:20split_2:output:21split_2:output:21split_2:output:22split_2:output:22split_2:output:23split_2:output:23split_2:output:24split_2:output:24split_2:output:25split_2:output:25split_2:output:26split_2:output:26split_2:output:27split_2:output:27split_2:output:28split_2:output:28split_2:output:29split_2:output:29split_2:output:30split_2:output:30split_2:output:31split_2:output:31concat_2/axis:output:0*
N@*
T0*5
_output_shapes#
!:’’’’’’’’’
@g
IdentityIdentityconcat_2:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’
@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :’’’’’’’’’@ :\ X
4
_output_shapes"
 :’’’’’’’’’@ 
 
_user_specified_nameinputs

¤
+__inference_conv3d_10_layer_call_fn_1044403

inputs%
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallī
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv3d_10_layer_call_and_return_conditional_losses_1041803}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’ 
 
_user_specified_nameinputs
æ
ä
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1043761

inputsC
%conv3d_conv3d_readvariableop_resource: 4
&conv3d_biasadd_readvariableop_resource: E
'conv3d_1_conv3d_readvariableop_resource:  6
(conv3d_1_biasadd_readvariableop_resource: 
identity¢conv3d/BiasAdd/ReadVariableOp¢conv3d/Conv3D/ReadVariableOp¢conv3d_1/BiasAdd/ReadVariableOp¢conv3d_1/Conv3D/ReadVariableOp
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0®
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	

conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ e
ReluReluconv3d/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0¾
conv3d_1/Conv3DConv3DRelu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	

conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ i
Relu_1Reluconv3d_1/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ q
IdentityIdentityRelu_1:activations:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ Č
NoOpNoOp^conv3d/BiasAdd/ReadVariableOp^conv3d/Conv3D/ReadVariableOp ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’: : : : 2>
conv3d/BiasAdd/ReadVariableOpconv3d/BiasAdd/ReadVariableOp2<
conv3d/Conv3D/ReadVariableOpconv3d/Conv3D/ReadVariableOp2B
conv3d_1/BiasAdd/ReadVariableOpconv3d_1/BiasAdd/ReadVariableOp2@
conv3d_1/Conv3D/ReadVariableOpconv3d_1/Conv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’
 
_user_specified_nameinputs
Ģ


E__inference_conv3d_7_layer_call_and_return_conditional_losses_1044603

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’
@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:\ X
4
_output_shapes"
 :’’’’’’’’’
@@
 
_user_specified_nameinputs
č
M
1__inference_up_sampling3d_1_layer_call_fn_1044254

inputs
identityŹ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’
@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_1042031n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’
@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :’’’’’’’’’@ :\ X
4
_output_shapes"
 :’’’’’’’’’@ 
 
_user_specified_nameinputs
×
f
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_1041771

inputs
identity½
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
ksize	
*
paddingSAME*
strides	

IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’: {
W
_output_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ų
Ö
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1041426
input_1/
conv3d_8_1041413:@
conv3d_8_1041415:	0
conv3d_9_1041419:
conv3d_9_1041421:	
identity¢ conv3d_8/StatefulPartitionedCall¢ conv3d_9/StatefulPartitionedCall
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_8_1041413conv3d_8_1041415*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_8_layer_call_and_return_conditional_losses_1041299v
ReluRelu)conv3d_8/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ 
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_9_1041419conv3d_9_1041421*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_9_layer_call_and_return_conditional_losses_1041316x
Relu_1Relu)conv3d_9/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ p
IdentityIdentityRelu_1:activations:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’@ 
NoOpNoOp!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’@ @: : : : 2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall:\ X
3
_output_shapes!
:’’’’’’’’’@ @
!
_user_specified_name	input_1
Žy
f
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_1044249

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :£
splitSplitsplit/split_dim:output:0inputs*
T0*Ö
_output_shapesĆ
Ą:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@*
	num_split
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :°
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9concat/axis:output:0*
N*
T0*4
_output_shapes"
 :’’’’’’’’’@@S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ń
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*
_output_shapes
:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@:’’’’’’’’’@@*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :'
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15split_1:output:16split_1:output:16split_1:output:17split_1:output:17split_1:output:18split_1:output:18split_1:output:19split_1:output:19split_1:output:20split_1:output:20split_1:output:21split_1:output:21split_1:output:22split_1:output:22split_1:output:23split_1:output:23split_1:output:24split_1:output:24split_1:output:25split_1:output:25split_1:output:26split_1:output:26split_1:output:27split_1:output:27split_1:output:28split_1:output:28split_1:output:29split_1:output:29split_1:output:30split_1:output:30split_1:output:31split_1:output:31split_1:output:32split_1:output:32split_1:output:33split_1:output:33split_1:output:34split_1:output:34split_1:output:35split_1:output:35split_1:output:36split_1:output:36split_1:output:37split_1:output:37split_1:output:38split_1:output:38split_1:output:39split_1:output:39split_1:output:40split_1:output:40split_1:output:41split_1:output:41split_1:output:42split_1:output:42split_1:output:43split_1:output:43split_1:output:44split_1:output:44split_1:output:45split_1:output:45split_1:output:46split_1:output:46split_1:output:47split_1:output:47split_1:output:48split_1:output:48split_1:output:49split_1:output:49split_1:output:50split_1:output:50split_1:output:51split_1:output:51split_1:output:52split_1:output:52split_1:output:53split_1:output:53split_1:output:54split_1:output:54split_1:output:55split_1:output:55split_1:output:56split_1:output:56split_1:output:57split_1:output:57split_1:output:58split_1:output:58split_1:output:59split_1:output:59split_1:output:60split_1:output:60split_1:output:61split_1:output:61split_1:output:62split_1:output:62split_1:output:63split_1:output:63split_1:output:64split_1:output:64split_1:output:65split_1:output:65split_1:output:66split_1:output:66split_1:output:67split_1:output:67split_1:output:68split_1:output:68split_1:output:69split_1:output:69split_1:output:70split_1:output:70split_1:output:71split_1:output:71split_1:output:72split_1:output:72split_1:output:73split_1:output:73split_1:output:74split_1:output:74split_1:output:75split_1:output:75split_1:output:76split_1:output:76split_1:output:77split_1:output:77split_1:output:78split_1:output:78split_1:output:79split_1:output:79split_1:output:80split_1:output:80split_1:output:81split_1:output:81split_1:output:82split_1:output:82split_1:output:83split_1:output:83split_1:output:84split_1:output:84split_1:output:85split_1:output:85split_1:output:86split_1:output:86split_1:output:87split_1:output:87split_1:output:88split_1:output:88split_1:output:89split_1:output:89split_1:output:90split_1:output:90split_1:output:91split_1:output:91split_1:output:92split_1:output:92split_1:output:93split_1:output:93split_1:output:94split_1:output:94split_1:output:95split_1:output:95split_1:output:96split_1:output:96split_1:output:97split_1:output:97split_1:output:98split_1:output:98split_1:output:99split_1:output:99split_1:output:100split_1:output:100split_1:output:101split_1:output:101split_1:output:102split_1:output:102split_1:output:103split_1:output:103split_1:output:104split_1:output:104split_1:output:105split_1:output:105split_1:output:106split_1:output:106split_1:output:107split_1:output:107split_1:output:108split_1:output:108split_1:output:109split_1:output:109split_1:output:110split_1:output:110split_1:output:111split_1:output:111split_1:output:112split_1:output:112split_1:output:113split_1:output:113split_1:output:114split_1:output:114split_1:output:115split_1:output:115split_1:output:116split_1:output:116split_1:output:117split_1:output:117split_1:output:118split_1:output:118split_1:output:119split_1:output:119split_1:output:120split_1:output:120split_1:output:121split_1:output:121split_1:output:122split_1:output:122split_1:output:123split_1:output:123split_1:output:124split_1:output:124split_1:output:125split_1:output:125split_1:output:126split_1:output:126split_1:output:127split_1:output:127concat_1/axis:output:0*
N*
T0*4
_output_shapes"
 :’’’’’’’’’@@S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ņ
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*
_output_shapes
:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@:’’’’’’’’’@*
	num_split@O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :ā
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15split_2:output:16split_2:output:16split_2:output:17split_2:output:17split_2:output:18split_2:output:18split_2:output:19split_2:output:19split_2:output:20split_2:output:20split_2:output:21split_2:output:21split_2:output:22split_2:output:22split_2:output:23split_2:output:23split_2:output:24split_2:output:24split_2:output:25split_2:output:25split_2:output:26split_2:output:26split_2:output:27split_2:output:27split_2:output:28split_2:output:28split_2:output:29split_2:output:29split_2:output:30split_2:output:30split_2:output:31split_2:output:31split_2:output:32split_2:output:32split_2:output:33split_2:output:33split_2:output:34split_2:output:34split_2:output:35split_2:output:35split_2:output:36split_2:output:36split_2:output:37split_2:output:37split_2:output:38split_2:output:38split_2:output:39split_2:output:39split_2:output:40split_2:output:40split_2:output:41split_2:output:41split_2:output:42split_2:output:42split_2:output:43split_2:output:43split_2:output:44split_2:output:44split_2:output:45split_2:output:45split_2:output:46split_2:output:46split_2:output:47split_2:output:47split_2:output:48split_2:output:48split_2:output:49split_2:output:49split_2:output:50split_2:output:50split_2:output:51split_2:output:51split_2:output:52split_2:output:52split_2:output:53split_2:output:53split_2:output:54split_2:output:54split_2:output:55split_2:output:55split_2:output:56split_2:output:56split_2:output:57split_2:output:57split_2:output:58split_2:output:58split_2:output:59split_2:output:59split_2:output:60split_2:output:60split_2:output:61split_2:output:61split_2:output:62split_2:output:62split_2:output:63split_2:output:63concat_2/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’@g
IdentityIdentityconcat_2:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :’’’’’’’’’
@@:\ X
4
_output_shapes"
 :’’’’’’’’’
@@
 
_user_specified_nameinputs
Ņ


F__inference_conv3d_10_layer_call_and_return_conditional_losses_1041803

inputs<
conv3d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype0 
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’ 
 
_user_specified_nameinputs

£
*__inference_conv3d_2_layer_call_fn_1044536

inputs%
unknown:` 
	unknown_0: 
identity¢StatefulPartitionedCallķ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_2_layer_call_and_return_conditional_losses_1041459}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’`: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’`
 
_user_specified_nameinputs
æ
ä
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1043743

inputsC
%conv3d_conv3d_readvariableop_resource: 4
&conv3d_biasadd_readvariableop_resource: E
'conv3d_1_conv3d_readvariableop_resource:  6
(conv3d_1_biasadd_readvariableop_resource: 
identity¢conv3d/BiasAdd/ReadVariableOp¢conv3d/Conv3D/ReadVariableOp¢conv3d_1/BiasAdd/ReadVariableOp¢conv3d_1/Conv3D/ReadVariableOp
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0®
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	

conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ e
ReluReluconv3d/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0¾
conv3d_1/Conv3DConv3DRelu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	

conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ i
Relu_1Reluconv3d_1/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ q
IdentityIdentityRelu_1:activations:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ Č
NoOpNoOp^conv3d/BiasAdd/ReadVariableOp^conv3d/Conv3D/ReadVariableOp ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’: : : : 2>
conv3d/BiasAdd/ReadVariableOpconv3d/BiasAdd/ReadVariableOp2<
conv3d/Conv3D/ReadVariableOpconv3d/Conv3D/ReadVariableOp2B
conv3d_1/BiasAdd/ReadVariableOpconv3d_1/BiasAdd/ReadVariableOp2@
conv3d_1/Conv3D/ReadVariableOpconv3d_1/Conv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’
 
_user_specified_nameinputs
Ņ


F__inference_conv3d_10_layer_call_and_return_conditional_losses_1044413

inputs<
conv3d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype0 
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’ 
 
_user_specified_nameinputs
é
ī
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1043947

inputsE
'conv3d_2_conv3d_readvariableop_resource:` 6
(conv3d_2_biasadd_readvariableop_resource: E
'conv3d_3_conv3d_readvariableop_resource:  6
(conv3d_3_biasadd_readvariableop_resource: 
identity¢conv3d_2/BiasAdd/ReadVariableOp¢conv3d_2/Conv3D/ReadVariableOp¢conv3d_3/BiasAdd/ReadVariableOp¢conv3d_3/Conv3D/ReadVariableOp
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:` *
dtype0²
conv3d_2/Conv3DConv3Dinputs&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	

conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ g
ReluReluconv3d_2/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0¾
conv3d_3/Conv3DConv3DRelu:activations:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	

conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ i
Relu_1Reluconv3d_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ q
IdentityIdentityRelu_1:activations:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ Ģ
NoOpNoOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’`: : : : 2B
conv3d_2/BiasAdd/ReadVariableOpconv3d_2/BiasAdd/ReadVariableOp2@
conv3d_2/Conv3D/ReadVariableOpconv3d_2/Conv3D/ReadVariableOp2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’`
 
_user_specified_nameinputs
”
°
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1043689

inputsF
(conv3d_10_conv3d_readvariableop_resource: 7
)conv3d_10_biasadd_readvariableop_resource:
identity¢ conv3d_10/BiasAdd/ReadVariableOp¢conv3d_10/Conv3D/ReadVariableOp
conv3d_10/Conv3D/ReadVariableOpReadVariableOp(conv3d_10_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0“
conv3d_10/Conv3DConv3Dinputs'conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’*
paddingSAME*
strides	

 conv3d_10/BiasAdd/ReadVariableOpReadVariableOp)conv3d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0”
conv3d_10/BiasAddBiasAddconv3d_10/Conv3D:output:0(conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’w
IdentityIdentityconv3d_10/BiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’
NoOpNoOp!^conv3d_10/BiasAdd/ReadVariableOp ^conv3d_10/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 2D
 conv3d_10/BiasAdd/ReadVariableOp conv3d_10/BiasAdd/ReadVariableOp2B
conv3d_10/Conv3D/ReadVariableOpconv3d_10/Conv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’ 
 
_user_specified_nameinputs
Ļ


C__inference_conv3d_layer_call_and_return_conditional_losses_1044432

inputs<
conv3d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype0 
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’
 
_user_specified_nameinputs
ų
ģ
0__inference_conv_block3d_2_layer_call_fn_1041175
input_1%
unknown: @
	unknown_0:@'
	unknown_1:@@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1041164|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:’’’’’’’’’
@ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :’’’’’’’’’
@ 
!
_user_specified_name	input_1
Ń


E__inference_conv3d_2_layer_call_and_return_conditional_losses_1041459

inputs<
conv3d_readvariableop_resource:` -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:` *
dtype0 
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:’’’’’’’’’ m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’`
 
_user_specified_nameinputs

Ŗ
0__inference_conv_block3d_5_layer_call_fn_1041817
input_1%
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallō
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1041810}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’ 
!
_user_specified_name	input_1
Ü
Ń
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1041586
input_1.
conv3d_2_1041573:` 
conv3d_2_1041575: .
conv3d_3_1041579:  
conv3d_3_1041581: 
identity¢ conv3d_2/StatefulPartitionedCall¢ conv3d_3/StatefulPartitionedCall
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_2_1041573conv3d_2_1041575*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_2_layer_call_and_return_conditional_losses_1041459w
ReluRelu)conv3d_2/StatefulPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_3_1041579conv3d_3_1041581*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_3_layer_call_and_return_conditional_losses_1041476y
Relu_1Relu)conv3d_3/StatefulPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ q
IdentityIdentityRelu_1:activations:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ 
NoOpNoOp!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’`: : : : 2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’`
!
_user_specified_name	input_1
ū
ń
0__inference_conv_block3d_4_layer_call_fn_1041335
input_1&
unknown:@
	unknown_0:	)
	unknown_1:
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’@ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1041324|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’@ @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:’’’’’’’’’@ @
!
_user_specified_name	input_1
Ņ
Š
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1041164

inputs.
conv3d_4_1041140: @
conv3d_4_1041142:@.
conv3d_5_1041157:@@
conv3d_5_1041159:@
identity¢ conv3d_4/StatefulPartitionedCall¢ conv3d_5/StatefulPartitionedCall
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_4_1041140conv3d_4_1041142*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_4_layer_call_and_return_conditional_losses_1041139v
ReluRelu)conv3d_4/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_5_1041157conv3d_5_1041159*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_5_layer_call_and_return_conditional_losses_1041156x
Relu_1Relu)conv3d_5/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@p
IdentityIdentityRelu_1:activations:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
NoOpNoOp!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:’’’’’’’’’
@ : : : : 2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall:\ X
4
_output_shapes"
 :’’’’’’’’’
@ 
 
_user_specified_nameinputs
	
ą
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1041847

inputs/
conv3d_10_1041841: 
conv3d_10_1041843:
identity¢!conv3d_10/StatefulPartitionedCall
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_10_1041841conv3d_10_1041843*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv3d_10_layer_call_and_return_conditional_losses_1041803
IdentityIdentity*conv3d_10/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’j
NoOpNoOp"^conv3d_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’ 
 
_user_specified_nameinputs

©
0__inference_conv_block3d_5_layer_call_fn_1043679

inputs%
unknown: 
	unknown_0:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1041847}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’ : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’ 
 
_user_specified_nameinputs
ų
ģ
0__inference_conv_block3d_3_layer_call_fn_1043973

inputs&
unknown:Ą@
	unknown_0:@'
	unknown_1:@@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1041706|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’
@Ą: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’
@Ą
 
_user_specified_nameinputs
Ī


E__inference_conv3d_8_layer_call_and_return_conditional_losses_1044508

inputs=
conv3d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ *
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’@ @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:’’’’’’’’’@ @
 
_user_specified_nameinputs
ź
Č
)__inference_u_net3d_layer_call_fn_1042339
input_1%
unknown: 
	unknown_0: '
	unknown_1:  
	unknown_2: '
	unknown_3: @
	unknown_4:@'
	unknown_5:@@
	unknown_6:@(
	unknown_7:@
	unknown_8:	)
	unknown_9:

unknown_10:	)

unknown_11:Ą@

unknown_12:@(

unknown_13:@@

unknown_14:@(

unknown_15:` 

unknown_16: (

unknown_17:  

unknown_18: (

unknown_19: 

unknown_20:
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_u_net3d_layer_call_and_return_conditional_losses_1042292}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’
!
_user_specified_name	input_1
¬
7
#__inference__traced_restore_1045074
file_prefixI
+assignvariableop_conv_block3d_conv3d_kernel: 9
+assignvariableop_1_conv_block3d_conv3d_bias: M
/assignvariableop_2_conv_block3d_conv3d_1_kernel:  ;
-assignvariableop_3_conv_block3d_conv3d_1_bias: O
1assignvariableop_4_conv_block3d_2_conv3d_4_kernel: @=
/assignvariableop_5_conv_block3d_2_conv3d_4_bias:@O
1assignvariableop_6_conv_block3d_2_conv3d_5_kernel:@@=
/assignvariableop_7_conv_block3d_2_conv3d_5_bias:@P
1assignvariableop_8_conv_block3d_4_conv3d_8_kernel:@>
/assignvariableop_9_conv_block3d_4_conv3d_8_bias:	R
2assignvariableop_10_conv_block3d_4_conv3d_9_kernel:?
0assignvariableop_11_conv_block3d_4_conv3d_9_bias:	P
2assignvariableop_12_conv_block3d_1_conv3d_2_kernel:` >
0assignvariableop_13_conv_block3d_1_conv3d_2_bias: P
2assignvariableop_14_conv_block3d_1_conv3d_3_kernel:  >
0assignvariableop_15_conv_block3d_1_conv3d_3_bias: Q
2assignvariableop_16_conv_block3d_3_conv3d_6_kernel:Ą@>
0assignvariableop_17_conv_block3d_3_conv3d_6_bias:@P
2assignvariableop_18_conv_block3d_3_conv3d_7_kernel:@@>
0assignvariableop_19_conv_block3d_3_conv3d_7_bias:@Q
3assignvariableop_20_conv_block3d_5_conv3d_10_kernel: ?
1assignvariableop_21_conv_block3d_5_conv3d_10_bias:'
assignvariableop_22_adam_iter:	 )
assignvariableop_23_adam_beta_1: )
assignvariableop_24_adam_beta_2: (
assignvariableop_25_adam_decay: 0
&assignvariableop_26_adam_learning_rate: #
assignvariableop_27_total: #
assignvariableop_28_count: S
5assignvariableop_29_adam_conv_block3d_conv3d_kernel_m: A
3assignvariableop_30_adam_conv_block3d_conv3d_bias_m: U
7assignvariableop_31_adam_conv_block3d_conv3d_1_kernel_m:  C
5assignvariableop_32_adam_conv_block3d_conv3d_1_bias_m: W
9assignvariableop_33_adam_conv_block3d_2_conv3d_4_kernel_m: @E
7assignvariableop_34_adam_conv_block3d_2_conv3d_4_bias_m:@W
9assignvariableop_35_adam_conv_block3d_2_conv3d_5_kernel_m:@@E
7assignvariableop_36_adam_conv_block3d_2_conv3d_5_bias_m:@X
9assignvariableop_37_adam_conv_block3d_4_conv3d_8_kernel_m:@F
7assignvariableop_38_adam_conv_block3d_4_conv3d_8_bias_m:	Y
9assignvariableop_39_adam_conv_block3d_4_conv3d_9_kernel_m:F
7assignvariableop_40_adam_conv_block3d_4_conv3d_9_bias_m:	W
9assignvariableop_41_adam_conv_block3d_1_conv3d_2_kernel_m:` E
7assignvariableop_42_adam_conv_block3d_1_conv3d_2_bias_m: W
9assignvariableop_43_adam_conv_block3d_1_conv3d_3_kernel_m:  E
7assignvariableop_44_adam_conv_block3d_1_conv3d_3_bias_m: X
9assignvariableop_45_adam_conv_block3d_3_conv3d_6_kernel_m:Ą@E
7assignvariableop_46_adam_conv_block3d_3_conv3d_6_bias_m:@W
9assignvariableop_47_adam_conv_block3d_3_conv3d_7_kernel_m:@@E
7assignvariableop_48_adam_conv_block3d_3_conv3d_7_bias_m:@X
:assignvariableop_49_adam_conv_block3d_5_conv3d_10_kernel_m: F
8assignvariableop_50_adam_conv_block3d_5_conv3d_10_bias_m:S
5assignvariableop_51_adam_conv_block3d_conv3d_kernel_v: A
3assignvariableop_52_adam_conv_block3d_conv3d_bias_v: U
7assignvariableop_53_adam_conv_block3d_conv3d_1_kernel_v:  C
5assignvariableop_54_adam_conv_block3d_conv3d_1_bias_v: W
9assignvariableop_55_adam_conv_block3d_2_conv3d_4_kernel_v: @E
7assignvariableop_56_adam_conv_block3d_2_conv3d_4_bias_v:@W
9assignvariableop_57_adam_conv_block3d_2_conv3d_5_kernel_v:@@E
7assignvariableop_58_adam_conv_block3d_2_conv3d_5_bias_v:@X
9assignvariableop_59_adam_conv_block3d_4_conv3d_8_kernel_v:@F
7assignvariableop_60_adam_conv_block3d_4_conv3d_8_bias_v:	Y
9assignvariableop_61_adam_conv_block3d_4_conv3d_9_kernel_v:F
7assignvariableop_62_adam_conv_block3d_4_conv3d_9_bias_v:	W
9assignvariableop_63_adam_conv_block3d_1_conv3d_2_kernel_v:` E
7assignvariableop_64_adam_conv_block3d_1_conv3d_2_bias_v: W
9assignvariableop_65_adam_conv_block3d_1_conv3d_3_kernel_v:  E
7assignvariableop_66_adam_conv_block3d_1_conv3d_3_bias_v: X
9assignvariableop_67_adam_conv_block3d_3_conv3d_6_kernel_v:Ą@E
7assignvariableop_68_adam_conv_block3d_3_conv3d_6_bias_v:@W
9assignvariableop_69_adam_conv_block3d_3_conv3d_7_kernel_v:@@E
7assignvariableop_70_adam_conv_block3d_3_conv3d_7_bias_v:@X
:assignvariableop_71_adam_conv_block3d_5_conv3d_10_kernel_v: F
8assignvariableop_72_adam_conv_block3d_5_conv3d_10_bias_v:
identity_74¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_8¢AssignVariableOp_9¢"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*Č!
value¾!B»!JB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*©
valueBJB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¾
_output_shapes«
Ø::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*X
dtypesN
L2J	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp+assignvariableop_conv_block3d_conv3d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp+assignvariableop_1_conv_block3d_conv3d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp/assignvariableop_2_conv_block3d_conv3d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp-assignvariableop_3_conv_block3d_conv3d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_4AssignVariableOp1assignvariableop_4_conv_block3d_2_conv3d_4_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp/assignvariableop_5_conv_block3d_2_conv3d_4_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_6AssignVariableOp1assignvariableop_6_conv_block3d_2_conv3d_5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp/assignvariableop_7_conv_block3d_2_conv3d_5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_8AssignVariableOp1assignvariableop_8_conv_block3d_4_conv3d_8_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp/assignvariableop_9_conv_block3d_4_conv3d_8_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_10AssignVariableOp2assignvariableop_10_conv_block3d_4_conv3d_9_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_11AssignVariableOp0assignvariableop_11_conv_block3d_4_conv3d_9_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_12AssignVariableOp2assignvariableop_12_conv_block3d_1_conv3d_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_13AssignVariableOp0assignvariableop_13_conv_block3d_1_conv3d_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_14AssignVariableOp2assignvariableop_14_conv_block3d_1_conv3d_3_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_15AssignVariableOp0assignvariableop_15_conv_block3d_1_conv3d_3_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_16AssignVariableOp2assignvariableop_16_conv_block3d_3_conv3d_6_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_17AssignVariableOp0assignvariableop_17_conv_block3d_3_conv3d_6_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_18AssignVariableOp2assignvariableop_18_conv_block3d_3_conv3d_7_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_19AssignVariableOp0assignvariableop_19_conv_block3d_3_conv3d_7_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_20AssignVariableOp3assignvariableop_20_conv_block3d_5_conv3d_10_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_21AssignVariableOp1assignvariableop_21_conv_block3d_5_conv3d_10_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_beta_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_beta_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_decayIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_learning_rateIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_29AssignVariableOp5assignvariableop_29_adam_conv_block3d_conv3d_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_30AssignVariableOp3assignvariableop_30_adam_conv_block3d_conv3d_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_31AssignVariableOp7assignvariableop_31_adam_conv_block3d_conv3d_1_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_32AssignVariableOp5assignvariableop_32_adam_conv_block3d_conv3d_1_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_33AssignVariableOp9assignvariableop_33_adam_conv_block3d_2_conv3d_4_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_34AssignVariableOp7assignvariableop_34_adam_conv_block3d_2_conv3d_4_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_35AssignVariableOp9assignvariableop_35_adam_conv_block3d_2_conv3d_5_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_36AssignVariableOp7assignvariableop_36_adam_conv_block3d_2_conv3d_5_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_37AssignVariableOp9assignvariableop_37_adam_conv_block3d_4_conv3d_8_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_38AssignVariableOp7assignvariableop_38_adam_conv_block3d_4_conv3d_8_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_39AssignVariableOp9assignvariableop_39_adam_conv_block3d_4_conv3d_9_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_40AssignVariableOp7assignvariableop_40_adam_conv_block3d_4_conv3d_9_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_41AssignVariableOp9assignvariableop_41_adam_conv_block3d_1_conv3d_2_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_42AssignVariableOp7assignvariableop_42_adam_conv_block3d_1_conv3d_2_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_43AssignVariableOp9assignvariableop_43_adam_conv_block3d_1_conv3d_3_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_44AssignVariableOp7assignvariableop_44_adam_conv_block3d_1_conv3d_3_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_45AssignVariableOp9assignvariableop_45_adam_conv_block3d_3_conv3d_6_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_46AssignVariableOp7assignvariableop_46_adam_conv_block3d_3_conv3d_6_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_47AssignVariableOp9assignvariableop_47_adam_conv_block3d_3_conv3d_7_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_48AssignVariableOp7assignvariableop_48_adam_conv_block3d_3_conv3d_7_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_49AssignVariableOp:assignvariableop_49_adam_conv_block3d_5_conv3d_10_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_50AssignVariableOp8assignvariableop_50_adam_conv_block3d_5_conv3d_10_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_51AssignVariableOp5assignvariableop_51_adam_conv_block3d_conv3d_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_52AssignVariableOp3assignvariableop_52_adam_conv_block3d_conv3d_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_53AssignVariableOp7assignvariableop_53_adam_conv_block3d_conv3d_1_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_54AssignVariableOp5assignvariableop_54_adam_conv_block3d_conv3d_1_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_55AssignVariableOp9assignvariableop_55_adam_conv_block3d_2_conv3d_4_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_56AssignVariableOp7assignvariableop_56_adam_conv_block3d_2_conv3d_4_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_57AssignVariableOp9assignvariableop_57_adam_conv_block3d_2_conv3d_5_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_58AssignVariableOp7assignvariableop_58_adam_conv_block3d_2_conv3d_5_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_59AssignVariableOp9assignvariableop_59_adam_conv_block3d_4_conv3d_8_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_60AssignVariableOp7assignvariableop_60_adam_conv_block3d_4_conv3d_8_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_61AssignVariableOp9assignvariableop_61_adam_conv_block3d_4_conv3d_9_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_62AssignVariableOp7assignvariableop_62_adam_conv_block3d_4_conv3d_9_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_63AssignVariableOp9assignvariableop_63_adam_conv_block3d_1_conv3d_2_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_64AssignVariableOp7assignvariableop_64_adam_conv_block3d_1_conv3d_2_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_65AssignVariableOp9assignvariableop_65_adam_conv_block3d_1_conv3d_3_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_66AssignVariableOp7assignvariableop_66_adam_conv_block3d_1_conv3d_3_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_67AssignVariableOp9assignvariableop_67_adam_conv_block3d_3_conv3d_6_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_68AssignVariableOp7assignvariableop_68_adam_conv_block3d_3_conv3d_6_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_69AssignVariableOp9assignvariableop_69_adam_conv_block3d_3_conv3d_7_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_70AssignVariableOp7assignvariableop_70_adam_conv_block3d_3_conv3d_7_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_71AssignVariableOp:assignvariableop_71_adam_conv_block3d_5_conv3d_10_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_72AssignVariableOp8assignvariableop_72_adam_conv_block3d_5_conv3d_10_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_73Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_74IdentityIdentity_73:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_74Identity_74:output:0*©
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ų
Ö
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1041442
input_1/
conv3d_8_1041429:@
conv3d_8_1041431:	0
conv3d_9_1041435:
conv3d_9_1041437:	
identity¢ conv3d_8/StatefulPartitionedCall¢ conv3d_9/StatefulPartitionedCall
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_8_1041429conv3d_8_1041431*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_8_layer_call_and_return_conditional_losses_1041299v
ReluRelu)conv3d_8/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ 
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_9_1041435conv3d_9_1041437*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_9_layer_call_and_return_conditional_losses_1041316x
Relu_1Relu)conv3d_9/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ p
IdentityIdentityRelu_1:activations:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’@ 
NoOpNoOp!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’@ @: : : : 2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall:\ X
3
_output_shapes!
:’’’’’’’’’@ @
!
_user_specified_name	input_1
Õ
Ń
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1041282
input_1.
conv3d_4_1041269: @
conv3d_4_1041271:@.
conv3d_5_1041275:@@
conv3d_5_1041277:@
identity¢ conv3d_4/StatefulPartitionedCall¢ conv3d_5/StatefulPartitionedCall
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_4_1041269conv3d_4_1041271*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_4_layer_call_and_return_conditional_losses_1041139v
ReluRelu)conv3d_4/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_5_1041275conv3d_5_1041277*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_5_layer_call_and_return_conditional_losses_1041156x
Relu_1Relu)conv3d_5/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@p
IdentityIdentityRelu_1:activations:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
NoOpNoOp!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:’’’’’’’’’
@ : : : : 2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall:] Y
4
_output_shapes"
 :’’’’’’’’’
@ 
!
_user_specified_name	input_1

¦
*__inference_conv3d_9_layer_call_fn_1044517

inputs'
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallģ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_9_layer_call_and_return_conditional_losses_1041316|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :’’’’’’’’’@ 
 
_user_specified_nameinputs
ä
ļ
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1043991

inputsF
'conv3d_6_conv3d_readvariableop_resource:Ą@6
(conv3d_6_biasadd_readvariableop_resource:@E
'conv3d_7_conv3d_readvariableop_resource:@@6
(conv3d_7_biasadd_readvariableop_resource:@
identity¢conv3d_6/BiasAdd/ReadVariableOp¢conv3d_6/Conv3D/ReadVariableOp¢conv3d_7/BiasAdd/ReadVariableOp¢conv3d_7/Conv3D/ReadVariableOp
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource*+
_output_shapes
:Ą@*
dtype0±
conv3d_6/Conv3DConv3Dinputs&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	

conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@f
ReluReluconv3d_6/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
conv3d_7/Conv3D/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0½
conv3d_7/Conv3DConv3DRelu:activations:0&conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	

conv3d_7/BiasAdd/ReadVariableOpReadVariableOp(conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv3d_7/BiasAddBiasAddconv3d_7/Conv3D:output:0'conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@h
Relu_1Reluconv3d_7/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@p
IdentityIdentityRelu_1:activations:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@Ģ
NoOpNoOp ^conv3d_6/BiasAdd/ReadVariableOp^conv3d_6/Conv3D/ReadVariableOp ^conv3d_7/BiasAdd/ReadVariableOp^conv3d_7/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’
@Ą: : : : 2B
conv3d_6/BiasAdd/ReadVariableOpconv3d_6/BiasAdd/ReadVariableOp2@
conv3d_6/Conv3D/ReadVariableOpconv3d_6/Conv3D/ReadVariableOp2B
conv3d_7/BiasAdd/ReadVariableOpconv3d_7/BiasAdd/ReadVariableOp2@
conv3d_7/Conv3D/ReadVariableOpconv3d_7/Conv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’
@Ą
 
_user_specified_nameinputs
ä
ļ
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1044009

inputsF
'conv3d_6_conv3d_readvariableop_resource:Ą@6
(conv3d_6_biasadd_readvariableop_resource:@E
'conv3d_7_conv3d_readvariableop_resource:@@6
(conv3d_7_biasadd_readvariableop_resource:@
identity¢conv3d_6/BiasAdd/ReadVariableOp¢conv3d_6/Conv3D/ReadVariableOp¢conv3d_7/BiasAdd/ReadVariableOp¢conv3d_7/Conv3D/ReadVariableOp
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource*+
_output_shapes
:Ą@*
dtype0±
conv3d_6/Conv3DConv3Dinputs&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	

conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@f
ReluReluconv3d_6/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
conv3d_7/Conv3D/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0½
conv3d_7/Conv3DConv3DRelu:activations:0&conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	

conv3d_7/BiasAdd/ReadVariableOpReadVariableOp(conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv3d_7/BiasAddBiasAddconv3d_7/Conv3D:output:0'conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@h
Relu_1Reluconv3d_7/BiasAdd:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@p
IdentityIdentityRelu_1:activations:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@Ģ
NoOpNoOp ^conv3d_6/BiasAdd/ReadVariableOp^conv3d_6/Conv3D/ReadVariableOp ^conv3d_7/BiasAdd/ReadVariableOp^conv3d_7/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’
@Ą: : : : 2B
conv3d_6/BiasAdd/ReadVariableOpconv3d_6/BiasAdd/ReadVariableOp2@
conv3d_6/Conv3D/ReadVariableOpconv3d_6/Conv3D/ReadVariableOp2B
conv3d_7/BiasAdd/ReadVariableOpconv3d_7/BiasAdd/ReadVariableOp2@
conv3d_7/Conv3D/ReadVariableOpconv3d_7/Conv3D/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’
@Ą
 
_user_specified_nameinputs
Ģ


E__inference_conv3d_7_layer_call_and_return_conditional_losses_1041636

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’
@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:\ X
4
_output_shapes"
 :’’’’’’’’’
@@
 
_user_specified_nameinputs

Y
-__inference_concatenate_layer_call_fn_1044374
inputs_0
inputs_1
identityÓ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1042275n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:’’’’’’’’’@:’’’’’’’’’ :_ [
5
_output_shapes#
!:’’’’’’’’’@
"
_user_specified_name
inputs/0:_[
5
_output_shapes#
!:’’’’’’’’’ 
"
_user_specified_name
inputs/1
Ģ


E__inference_conv3d_4_layer_call_and_return_conditional_losses_1044470

inputs<
conv3d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: @*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’
@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:\ X
4
_output_shapes"
 :’’’’’’’’’
@ 
 
_user_specified_nameinputs
Ü
Ń
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1041602
input_1.
conv3d_2_1041589:` 
conv3d_2_1041591: .
conv3d_3_1041595:  
conv3d_3_1041597: 
identity¢ conv3d_2/StatefulPartitionedCall¢ conv3d_3/StatefulPartitionedCall
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_2_1041589conv3d_2_1041591*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_2_layer_call_and_return_conditional_losses_1041459w
ReluRelu)conv3d_2/StatefulPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ 
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_3_1041595conv3d_3_1041597*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_3_layer_call_and_return_conditional_losses_1041476y
Relu_1Relu)conv3d_3/StatefulPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ q
IdentityIdentityRelu_1:activations:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ 
NoOpNoOp!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’`: : : : 2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall:^ Z
5
_output_shapes#
!:’’’’’’’’’`
!
_user_specified_name	input_1
Ł
h
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_1041783

inputs
identity½
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
ksize	
*
paddingSAME*
strides	

IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’: {
W
_output_shapesE
C:A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

„
*__inference_conv3d_8_layer_call_fn_1044498

inputs&
unknown:@
	unknown_0:	
identity¢StatefulPartitionedCallģ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_8_layer_call_and_return_conditional_losses_1041299|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’@ @: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:’’’’’’’’’@ @
 
_user_specified_nameinputs
Õ
Ń
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1041644

inputs/
conv3d_6_1041620:Ą@
conv3d_6_1041622:@.
conv3d_7_1041637:@@
conv3d_7_1041639:@
identity¢ conv3d_6/StatefulPartitionedCall¢ conv3d_7/StatefulPartitionedCall
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_6_1041620conv3d_6_1041622*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_6_layer_call_and_return_conditional_losses_1041619v
ReluRelu)conv3d_6/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_7_1041637conv3d_7_1041639*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_7_layer_call_and_return_conditional_losses_1041636x
Relu_1Relu)conv3d_7/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@p
IdentityIdentityRelu_1:activations:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
NoOpNoOp!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):’’’’’’’’’
@Ą: : : : 2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’
@Ą
 
_user_specified_nameinputs

”
(__inference_conv3d_layer_call_fn_1044422

inputs%
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallė
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_conv3d_layer_call_and_return_conditional_losses_1040979}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’
 
_user_specified_nameinputs
Ņ
Š
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1041226

inputs.
conv3d_4_1041213: @
conv3d_4_1041215:@.
conv3d_5_1041219:@@
conv3d_5_1041221:@
identity¢ conv3d_4/StatefulPartitionedCall¢ conv3d_5/StatefulPartitionedCall
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_4_1041213conv3d_4_1041215*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_4_layer_call_and_return_conditional_losses_1041139v
ReluRelu)conv3d_4/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv3d_5_1041219conv3d_5_1041221*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’
@@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv3d_5_layer_call_and_return_conditional_losses_1041156x
Relu_1Relu)conv3d_5/StatefulPartitionedCall:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’
@@p
IdentityIdentityRelu_1:activations:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’
@@
NoOpNoOp!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:’’’’’’’’’
@ : : : : 2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall:\ X
4
_output_shapes"
 :’’’’’’’’’
@ 
 
_user_specified_nameinputs

r
H__inference_concatenate_layer_call_and_return_conditional_losses_1042275

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*5
_output_shapes#
!:’’’’’’’’’`e
IdentityIdentityconcat:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:’’’’’’’’’@:’’’’’’’’’ :] Y
5
_output_shapes#
!:’’’’’’’’’@
 
_user_specified_nameinputs:]Y
5
_output_shapes#
!:’’’’’’’’’ 
 
_user_specified_nameinputs
Ņ


E__inference_conv3d_9_layer_call_and_return_conditional_losses_1041316

inputs>
conv3d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ *
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :’’’’’’’’’@ l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:\ X
4
_output_shapes"
 :’’’’’’’’’@ 
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ē
serving_default³
I
input_1>
serving_default_input_1:0’’’’’’’’’J
output_1>
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:Ī
Ś
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
_filters
	_dwt_kwargs

_enc_blocks
_dec_blocks

_pools
_upsamps
_concats

_out_block
	optimizer

signatures"
_tf_keras_model
Ę
0
1
2
3
4
5
6
7
8
9
10
11
12
13
 14
!15
"16
#17
$18
%19
&20
'21"
trackable_list_wrapper
Ę
0
1
2
3
4
5
6
7
8
9
10
11
12
13
 14
!15
"16
#17
$18
%19
&20
'21"
trackable_list_wrapper
 "
trackable_list_wrapper
Ź
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ķ
-trace_0
.trace_1
/trace_2
0trace_32ā
)__inference_u_net3d_layer_call_fn_1042339
)__inference_u_net3d_layer_call_fn_1042798
)__inference_u_net3d_layer_call_fn_1042847
)__inference_u_net3d_layer_call_fn_1042572³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z-trace_0z.trace_1z/trace_2z0trace_3
¹
1trace_0
2trace_1
3trace_2
4trace_32Ī
D__inference_u_net3d_layer_call_and_return_conditional_losses_1043254
D__inference_u_net3d_layer_call_and_return_conditional_losses_1043661
D__inference_u_net3d_layer_call_and_return_conditional_losses_1042632
D__inference_u_net3d_layer_call_and_return_conditional_losses_1042692³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z1trace_0z2trace_1z3trace_2z4trace_3
ĶBŹ
"__inference__wrapped_model_1040962input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
50
61
72"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
Ś
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
F_filters

G_convs

H_norms
I	_dropouts"
_tf_keras_model

Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratemČmÉmŹmĖmĢmĶmĪmĻmŠmŃmŅmÓmŌmÕ mÖ!m×"mŲ#mŁ$mŚ%mŪ&mÜ'mŻvŽvßvąvįvāvćvävåvęvēvčvévźvė vģ!vķ"vī#vļ$vš%vń&vņ'vó"
	optimizer
,
Oserving_default"
signature_map
8:6 2conv_block3d/conv3d/kernel
&:$ 2conv_block3d/conv3d/bias
::8  2conv_block3d/conv3d_1/kernel
(:& 2conv_block3d/conv3d_1/bias
<:: @2conv_block3d_2/conv3d_4/kernel
*:(@2conv_block3d_2/conv3d_4/bias
<::@@2conv_block3d_2/conv3d_5/kernel
*:(@2conv_block3d_2/conv3d_5/bias
=:;@2conv_block3d_4/conv3d_8/kernel
+:)2conv_block3d_4/conv3d_8/bias
>:<2conv_block3d_4/conv3d_9/kernel
+:)2conv_block3d_4/conv3d_9/bias
<::` 2conv_block3d_1/conv3d_2/kernel
*:( 2conv_block3d_1/conv3d_2/bias
<::  2conv_block3d_1/conv3d_3/kernel
*:( 2conv_block3d_1/conv3d_3/bias
=:;Ą@2conv_block3d_3/conv3d_6/kernel
*:(@2conv_block3d_3/conv3d_6/bias
<::@@2conv_block3d_3/conv3d_7/kernel
*:(@2conv_block3d_3/conv3d_7/bias
=:; 2conv_block3d_5/conv3d_10/kernel
+:)2conv_block3d_5/conv3d_10/bias
 "
trackable_list_wrapper
v
50
61
72
83
94
:5
;6
<7
=8
>9
?10
11"
trackable_list_wrapper
'
P0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ļBģ
)__inference_u_net3d_layer_call_fn_1042339input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
īBė
)__inference_u_net3d_layer_call_fn_1042798inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
īBė
)__inference_u_net3d_layer_call_fn_1042847inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ļBģ
)__inference_u_net3d_layer_call_fn_1042572input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
D__inference_u_net3d_layer_call_and_return_conditional_losses_1043254inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
D__inference_u_net3d_layer_call_and_return_conditional_losses_1043661inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
D__inference_u_net3d_layer_call_and_return_conditional_losses_1042632input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
D__inference_u_net3d_layer_call_and_return_conditional_losses_1042692input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ś
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
W_filters

X_convs

Y_norms
Z	_dropouts"
_tf_keras_model
Ś
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
a_filters

b_convs

c_norms
d	_dropouts"
_tf_keras_model
Ś
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
k_filters

l_convs

m_norms
n	_dropouts"
_tf_keras_model
Ś
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses
u_filters

v_convs

w_norms
x	_dropouts"
_tf_keras_model
Ż
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
_filters
_convs
_norms
	_dropouts"
_tf_keras_model
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
«
”	variables
¢trainable_variables
£regularization_losses
¤	keras_api
„__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_layer
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
§non_trainable_variables
Ølayers
©metrics
 Ŗlayer_regularization_losses
«layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
ń
¬trace_0
­trace_1
®trace_2
Ætrace_32ž
0__inference_conv_block3d_5_layer_call_fn_1041817
0__inference_conv_block3d_5_layer_call_fn_1043670
0__inference_conv_block3d_5_layer_call_fn_1043679
0__inference_conv_block3d_5_layer_call_fn_1041863³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z¬trace_0z­trace_1z®trace_2zÆtrace_3
Ż
°trace_0
±trace_1
²trace_2
³trace_32ź
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1043689
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1043699
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1041872
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1041881³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z°trace_0z±trace_1z²trace_2z³trace_3
 "
trackable_list_wrapper
(
“0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ĢBÉ
%__inference_signature_wrapper_1042749input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
R
µ	variables
¶	keras_api

·total

øcount"
_tf_keras_metric
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¹non_trainable_variables
ŗlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
é
¾trace_0
ætrace_1
Ątrace_2
Įtrace_32ö
.__inference_conv_block3d_layer_call_fn_1041015
.__inference_conv_block3d_layer_call_fn_1043712
.__inference_conv_block3d_layer_call_fn_1043725
.__inference_conv_block3d_layer_call_fn_1041090³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z¾trace_0zætrace_1zĄtrace_2zĮtrace_3
Õ
Ātrace_0
Ćtrace_1
Ätrace_2
Åtrace_32ā
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1043743
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1043761
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1041106
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1041122³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zĀtrace_0zĆtrace_1zÄtrace_2zÅtrace_3
 "
trackable_list_wrapper
0
Ę0
Ē1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Čnon_trainable_variables
Élayers
Źmetrics
 Ėlayer_regularization_losses
Ģlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
ń
Ķtrace_0
Ītrace_1
Ļtrace_2
Štrace_32ž
0__inference_conv_block3d_2_layer_call_fn_1041175
0__inference_conv_block3d_2_layer_call_fn_1043774
0__inference_conv_block3d_2_layer_call_fn_1043787
0__inference_conv_block3d_2_layer_call_fn_1041250³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zĶtrace_0zĪtrace_1zĻtrace_2zŠtrace_3
Ż
Ńtrace_0
Ņtrace_1
Ótrace_2
Ōtrace_32ź
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1043805
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1043823
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1041266
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1041282³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zŃtrace_0zŅtrace_1zÓtrace_2zŌtrace_3
 "
trackable_list_wrapper
0
Õ0
Ö1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
²
×non_trainable_variables
Ųlayers
Łmetrics
 Ślayer_regularization_losses
Ūlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
ń
Ütrace_0
Żtrace_1
Žtrace_2
ßtrace_32ž
0__inference_conv_block3d_4_layer_call_fn_1041335
0__inference_conv_block3d_4_layer_call_fn_1043836
0__inference_conv_block3d_4_layer_call_fn_1043849
0__inference_conv_block3d_4_layer_call_fn_1041410³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zÜtrace_0zŻtrace_1zŽtrace_2zßtrace_3
Ż
ątrace_0
įtrace_1
ātrace_2
ćtrace_32ź
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1043867
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1043885
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1041426
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1041442³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zątrace_0zįtrace_1zātrace_2zćtrace_3
 "
trackable_list_wrapper
0
ä0
å1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
 2
!3"
trackable_list_wrapper
<
0
1
 2
!3"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ęnon_trainable_variables
ēlayers
čmetrics
 élayer_regularization_losses
źlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
ń
ėtrace_0
ģtrace_1
ķtrace_2
ītrace_32ž
0__inference_conv_block3d_1_layer_call_fn_1041495
0__inference_conv_block3d_1_layer_call_fn_1043898
0__inference_conv_block3d_1_layer_call_fn_1043911
0__inference_conv_block3d_1_layer_call_fn_1041570³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zėtrace_0zģtrace_1zķtrace_2zītrace_3
Ż
ļtrace_0
štrace_1
ńtrace_2
ņtrace_32ź
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1043929
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1043947
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1041586
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1041602³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zļtrace_0zštrace_1zńtrace_2zņtrace_3
 "
trackable_list_wrapper
0
ó0
ō1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
"0
#1
$2
%3"
trackable_list_wrapper
<
"0
#1
$2
%3"
trackable_list_wrapper
 "
trackable_list_wrapper
²
õnon_trainable_variables
ölayers
÷metrics
 ųlayer_regularization_losses
łlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
ń
śtrace_0
ūtrace_1
ütrace_2
żtrace_32ž
0__inference_conv_block3d_3_layer_call_fn_1041655
0__inference_conv_block3d_3_layer_call_fn_1043960
0__inference_conv_block3d_3_layer_call_fn_1043973
0__inference_conv_block3d_3_layer_call_fn_1041730³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zśtrace_0zūtrace_1zütrace_2zżtrace_3
Ż
žtrace_0
’trace_1
trace_2
trace_32ź
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1043991
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1044009
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1041746
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1041762³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zžtrace_0z’trace_1ztrace_2ztrace_3
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
õ
trace_02Ö
/__inference_max_pooling3d_layer_call_fn_1044014¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0

trace_02ń
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_1044019¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
÷
trace_02Ų
1__inference_max_pooling3d_1_layer_call_fn_1044024¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0

trace_02ó
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_1044029¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
õ
trace_02Ö
/__inference_up_sampling3d_layer_call_fn_1044034¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0

trace_02ń
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_1044249¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
÷
trace_02Ų
1__inference_up_sampling3d_1_layer_call_fn_1044254¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0

trace_02ó
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_1044368¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
 non_trainable_variables
”layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
ó
„trace_02Ō
-__inference_concatenate_layer_call_fn_1044374¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z„trace_0

¦trace_02ļ
H__inference_concatenate_layer_call_and_return_conditional_losses_1044381¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z¦trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
§non_trainable_variables
Ølayers
©metrics
 Ŗlayer_regularization_losses
«layer_metrics
”	variables
¢trainable_variables
£regularization_losses
„__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
õ
¬trace_02Ö
/__inference_concatenate_1_layer_call_fn_1044387¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z¬trace_0

­trace_02ń
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1044394¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z­trace_0
 "
trackable_list_wrapper
(
“0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
öBó
0__inference_conv_block3d_5_layer_call_fn_1041817input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
õBņ
0__inference_conv_block3d_5_layer_call_fn_1043670inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
õBņ
0__inference_conv_block3d_5_layer_call_fn_1043679inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
öBó
0__inference_conv_block3d_5_layer_call_fn_1041863input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1043689inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1043699inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1041872input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1041881input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ä
®	variables
Ætrainable_variables
°regularization_losses
±	keras_api
²__call__
+³&call_and_return_all_conditional_losses

&kernel
'bias
!“_jit_compiled_convolution_op"
_tf_keras_layer
0
·0
ø1"
trackable_list_wrapper
.
µ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_list_wrapper
0
Ę0
Ē1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ōBń
.__inference_conv_block3d_layer_call_fn_1041015input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
óBš
.__inference_conv_block3d_layer_call_fn_1043712inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
óBš
.__inference_conv_block3d_layer_call_fn_1043725inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ōBń
.__inference_conv_block3d_layer_call_fn_1041090input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1043743inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1043761inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1041106input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1041122input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ä
µ	variables
¶trainable_variables
·regularization_losses
ø	keras_api
¹__call__
+ŗ&call_and_return_all_conditional_losses

kernel
bias
!»_jit_compiled_convolution_op"
_tf_keras_layer
ä
¼	variables
½trainable_variables
¾regularization_losses
æ	keras_api
Ą__call__
+Į&call_and_return_all_conditional_losses

kernel
bias
!Ā_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
0
Õ0
Ö1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
öBó
0__inference_conv_block3d_2_layer_call_fn_1041175input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
õBņ
0__inference_conv_block3d_2_layer_call_fn_1043774inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
õBņ
0__inference_conv_block3d_2_layer_call_fn_1043787inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
öBó
0__inference_conv_block3d_2_layer_call_fn_1041250input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1043805inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1043823inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1041266input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1041282input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ä
Ć	variables
Ätrainable_variables
Åregularization_losses
Ę	keras_api
Ē__call__
+Č&call_and_return_all_conditional_losses

kernel
bias
!É_jit_compiled_convolution_op"
_tf_keras_layer
ä
Ź	variables
Ėtrainable_variables
Ģregularization_losses
Ķ	keras_api
Ī__call__
+Ļ&call_and_return_all_conditional_losses

kernel
bias
!Š_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
0
ä0
å1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
öBó
0__inference_conv_block3d_4_layer_call_fn_1041335input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
õBņ
0__inference_conv_block3d_4_layer_call_fn_1043836inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
õBņ
0__inference_conv_block3d_4_layer_call_fn_1043849inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
öBó
0__inference_conv_block3d_4_layer_call_fn_1041410input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1043867inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1043885inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1041426input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1041442input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ä
Ń	variables
Ņtrainable_variables
Óregularization_losses
Ō	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses

kernel
bias
!×_jit_compiled_convolution_op"
_tf_keras_layer
ä
Ų	variables
Łtrainable_variables
Śregularization_losses
Ū	keras_api
Ü__call__
+Ż&call_and_return_all_conditional_losses

kernel
bias
!Ž_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
0
ó0
ō1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
öBó
0__inference_conv_block3d_1_layer_call_fn_1041495input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
õBņ
0__inference_conv_block3d_1_layer_call_fn_1043898inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
õBņ
0__inference_conv_block3d_1_layer_call_fn_1043911inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
öBó
0__inference_conv_block3d_1_layer_call_fn_1041570input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1043929inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1043947inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1041586input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1041602input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ä
ß	variables
ątrainable_variables
įregularization_losses
ā	keras_api
ć__call__
+ä&call_and_return_all_conditional_losses

kernel
bias
!å_jit_compiled_convolution_op"
_tf_keras_layer
ä
ę	variables
ētrainable_variables
čregularization_losses
é	keras_api
ź__call__
+ė&call_and_return_all_conditional_losses

 kernel
!bias
!ģ_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
öBó
0__inference_conv_block3d_3_layer_call_fn_1041655input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
õBņ
0__inference_conv_block3d_3_layer_call_fn_1043960inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
õBņ
0__inference_conv_block3d_3_layer_call_fn_1043973inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
öBó
0__inference_conv_block3d_3_layer_call_fn_1041730input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1043991inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1044009inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1041746input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1041762input_1"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ä
ķ	variables
ītrainable_variables
ļregularization_losses
š	keras_api
ń__call__
+ņ&call_and_return_all_conditional_losses

"kernel
#bias
!ó_jit_compiled_convolution_op"
_tf_keras_layer
ä
ō	variables
õtrainable_variables
öregularization_losses
÷	keras_api
ų__call__
+ł&call_and_return_all_conditional_losses

$kernel
%bias
!ś_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ćBą
/__inference_max_pooling3d_layer_call_fn_1044014inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
žBū
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_1044019inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
åBā
1__inference_max_pooling3d_1_layer_call_fn_1044024inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Bż
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_1044029inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ćBą
/__inference_up_sampling3d_layer_call_fn_1044034inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
žBū
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_1044249inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
åBā
1__inference_up_sampling3d_1_layer_call_fn_1044254inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Bż
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_1044368inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ķBź
-__inference_concatenate_layer_call_fn_1044374inputs/0inputs/1"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
H__inference_concatenate_layer_call_and_return_conditional_losses_1044381inputs/0inputs/1"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ļBģ
/__inference_concatenate_1_layer_call_fn_1044387inputs/0inputs/1"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1044394inputs/0inputs/1"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ūnon_trainable_variables
ülayers
żmetrics
 žlayer_regularization_losses
’layer_metrics
®	variables
Ætrainable_variables
°regularization_losses
²__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
ń
trace_02Ņ
+__inference_conv3d_10_layer_call_fn_1044403¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0

trace_02ķ
F__inference_conv3d_10_layer_call_and_return_conditional_losses_1044413¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0
“2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
µ	variables
¶trainable_variables
·regularization_losses
¹__call__
+ŗ&call_and_return_all_conditional_losses
'ŗ"call_and_return_conditional_losses"
_generic_user_object
ī
trace_02Ļ
(__inference_conv3d_layer_call_fn_1044422¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0

trace_02ź
C__inference_conv3d_layer_call_and_return_conditional_losses_1044432¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0
“2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¼	variables
½trainable_variables
¾regularization_losses
Ą__call__
+Į&call_and_return_all_conditional_losses
'Į"call_and_return_conditional_losses"
_generic_user_object
š
trace_02Ń
*__inference_conv3d_1_layer_call_fn_1044441¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0

trace_02ģ
E__inference_conv3d_1_layer_call_and_return_conditional_losses_1044451¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0
“2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ć	variables
Ätrainable_variables
Åregularization_losses
Ē__call__
+Č&call_and_return_all_conditional_losses
'Č"call_and_return_conditional_losses"
_generic_user_object
š
trace_02Ń
*__inference_conv3d_4_layer_call_fn_1044460¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0

trace_02ģ
E__inference_conv3d_4_layer_call_and_return_conditional_losses_1044470¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0
“2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ź	variables
Ėtrainable_variables
Ģregularization_losses
Ī__call__
+Ļ&call_and_return_all_conditional_losses
'Ļ"call_and_return_conditional_losses"
_generic_user_object
š
trace_02Ń
*__inference_conv3d_5_layer_call_fn_1044479¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0

trace_02ģ
E__inference_conv3d_5_layer_call_and_return_conditional_losses_1044489¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0
“2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
 metrics
 ”layer_regularization_losses
¢layer_metrics
Ń	variables
Ņtrainable_variables
Óregularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
š
£trace_02Ń
*__inference_conv3d_8_layer_call_fn_1044498¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z£trace_0

¤trace_02ģ
E__inference_conv3d_8_layer_call_and_return_conditional_losses_1044508¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z¤trace_0
“2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
„non_trainable_variables
¦layers
§metrics
 Ølayer_regularization_losses
©layer_metrics
Ų	variables
Łtrainable_variables
Śregularization_losses
Ü__call__
+Ż&call_and_return_all_conditional_losses
'Ż"call_and_return_conditional_losses"
_generic_user_object
š
Ŗtrace_02Ń
*__inference_conv3d_9_layer_call_fn_1044517¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zŖtrace_0

«trace_02ģ
E__inference_conv3d_9_layer_call_and_return_conditional_losses_1044527¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z«trace_0
“2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
¬non_trainable_variables
­layers
®metrics
 Ælayer_regularization_losses
°layer_metrics
ß	variables
ątrainable_variables
įregularization_losses
ć__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses"
_generic_user_object
š
±trace_02Ń
*__inference_conv3d_2_layer_call_fn_1044536¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z±trace_0

²trace_02ģ
E__inference_conv3d_2_layer_call_and_return_conditional_losses_1044546¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z²trace_0
“2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
³non_trainable_variables
“layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
ę	variables
ētrainable_variables
čregularization_losses
ź__call__
+ė&call_and_return_all_conditional_losses
'ė"call_and_return_conditional_losses"
_generic_user_object
š
øtrace_02Ń
*__inference_conv3d_3_layer_call_fn_1044555¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zøtrace_0

¹trace_02ģ
E__inference_conv3d_3_layer_call_and_return_conditional_losses_1044565¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z¹trace_0
“2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ŗnon_trainable_variables
»layers
¼metrics
 ½layer_regularization_losses
¾layer_metrics
ķ	variables
ītrainable_variables
ļregularization_losses
ń__call__
+ņ&call_and_return_all_conditional_losses
'ņ"call_and_return_conditional_losses"
_generic_user_object
š
ætrace_02Ń
*__inference_conv3d_6_layer_call_fn_1044574¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zætrace_0

Ątrace_02ģ
E__inference_conv3d_6_layer_call_and_return_conditional_losses_1044584¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zĄtrace_0
“2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Įnon_trainable_variables
Ālayers
Ćmetrics
 Älayer_regularization_losses
Ålayer_metrics
ō	variables
õtrainable_variables
öregularization_losses
ų__call__
+ł&call_and_return_all_conditional_losses
'ł"call_and_return_conditional_losses"
_generic_user_object
š
Ętrace_02Ń
*__inference_conv3d_7_layer_call_fn_1044593¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zĘtrace_0

Ētrace_02ģ
E__inference_conv3d_7_layer_call_and_return_conditional_losses_1044603¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zĒtrace_0
“2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_conv3d_10_layer_call_fn_1044403inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
śB÷
F__inference_conv3d_10_layer_call_and_return_conditional_losses_1044413inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÜBŁ
(__inference_conv3d_layer_call_fn_1044422inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
÷Bō
C__inference_conv3d_layer_call_and_return_conditional_losses_1044432inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŽBŪ
*__inference_conv3d_1_layer_call_fn_1044441inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
łBö
E__inference_conv3d_1_layer_call_and_return_conditional_losses_1044451inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŽBŪ
*__inference_conv3d_4_layer_call_fn_1044460inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
łBö
E__inference_conv3d_4_layer_call_and_return_conditional_losses_1044470inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŽBŪ
*__inference_conv3d_5_layer_call_fn_1044479inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
łBö
E__inference_conv3d_5_layer_call_and_return_conditional_losses_1044489inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŽBŪ
*__inference_conv3d_8_layer_call_fn_1044498inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
łBö
E__inference_conv3d_8_layer_call_and_return_conditional_losses_1044508inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŽBŪ
*__inference_conv3d_9_layer_call_fn_1044517inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
łBö
E__inference_conv3d_9_layer_call_and_return_conditional_losses_1044527inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŽBŪ
*__inference_conv3d_2_layer_call_fn_1044536inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
łBö
E__inference_conv3d_2_layer_call_and_return_conditional_losses_1044546inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŽBŪ
*__inference_conv3d_3_layer_call_fn_1044555inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
łBö
E__inference_conv3d_3_layer_call_and_return_conditional_losses_1044565inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŽBŪ
*__inference_conv3d_6_layer_call_fn_1044574inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
łBö
E__inference_conv3d_6_layer_call_and_return_conditional_losses_1044584inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŽBŪ
*__inference_conv3d_7_layer_call_fn_1044593inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
łBö
E__inference_conv3d_7_layer_call_and_return_conditional_losses_1044603inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
=:; 2!Adam/conv_block3d/conv3d/kernel/m
+:) 2Adam/conv_block3d/conv3d/bias/m
?:=  2#Adam/conv_block3d/conv3d_1/kernel/m
-:+ 2!Adam/conv_block3d/conv3d_1/bias/m
A:? @2%Adam/conv_block3d_2/conv3d_4/kernel/m
/:-@2#Adam/conv_block3d_2/conv3d_4/bias/m
A:?@@2%Adam/conv_block3d_2/conv3d_5/kernel/m
/:-@2#Adam/conv_block3d_2/conv3d_5/bias/m
B:@@2%Adam/conv_block3d_4/conv3d_8/kernel/m
0:.2#Adam/conv_block3d_4/conv3d_8/bias/m
C:A2%Adam/conv_block3d_4/conv3d_9/kernel/m
0:.2#Adam/conv_block3d_4/conv3d_9/bias/m
A:?` 2%Adam/conv_block3d_1/conv3d_2/kernel/m
/:- 2#Adam/conv_block3d_1/conv3d_2/bias/m
A:?  2%Adam/conv_block3d_1/conv3d_3/kernel/m
/:- 2#Adam/conv_block3d_1/conv3d_3/bias/m
B:@Ą@2%Adam/conv_block3d_3/conv3d_6/kernel/m
/:-@2#Adam/conv_block3d_3/conv3d_6/bias/m
A:?@@2%Adam/conv_block3d_3/conv3d_7/kernel/m
/:-@2#Adam/conv_block3d_3/conv3d_7/bias/m
B:@ 2&Adam/conv_block3d_5/conv3d_10/kernel/m
0:.2$Adam/conv_block3d_5/conv3d_10/bias/m
=:; 2!Adam/conv_block3d/conv3d/kernel/v
+:) 2Adam/conv_block3d/conv3d/bias/v
?:=  2#Adam/conv_block3d/conv3d_1/kernel/v
-:+ 2!Adam/conv_block3d/conv3d_1/bias/v
A:? @2%Adam/conv_block3d_2/conv3d_4/kernel/v
/:-@2#Adam/conv_block3d_2/conv3d_4/bias/v
A:?@@2%Adam/conv_block3d_2/conv3d_5/kernel/v
/:-@2#Adam/conv_block3d_2/conv3d_5/bias/v
B:@@2%Adam/conv_block3d_4/conv3d_8/kernel/v
0:.2#Adam/conv_block3d_4/conv3d_8/bias/v
C:A2%Adam/conv_block3d_4/conv3d_9/kernel/v
0:.2#Adam/conv_block3d_4/conv3d_9/bias/v
A:?` 2%Adam/conv_block3d_1/conv3d_2/kernel/v
/:- 2#Adam/conv_block3d_1/conv3d_2/bias/v
A:?  2%Adam/conv_block3d_1/conv3d_3/kernel/v
/:- 2#Adam/conv_block3d_1/conv3d_3/bias/v
B:@Ą@2%Adam/conv_block3d_3/conv3d_6/kernel/v
/:-@2#Adam/conv_block3d_3/conv3d_6/bias/v
A:?@@2%Adam/conv_block3d_3/conv3d_7/kernel/v
/:-@2#Adam/conv_block3d_3/conv3d_7/bias/v
B:@ 2&Adam/conv_block3d_5/conv3d_10/kernel/v
0:.2$Adam/conv_block3d_5/conv3d_10/bias/vĀ
"__inference__wrapped_model_1040962"#$% !&'>¢;
4¢1
/,
input_1’’’’’’’’’
Ŗ "AŖ>
<
output_10-
output_1’’’’’’’’’ū
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1044394¬u¢r
k¢h
fc
0-
inputs/0’’’’’’’’’
@
/,
inputs/1’’’’’’’’’
@@
Ŗ "3¢0
)&
0’’’’’’’’’
@Ą
 Ó
/__inference_concatenate_1_layer_call_fn_1044387u¢r
k¢h
fc
0-
inputs/0’’’’’’’’’
@
/,
inputs/1’’’’’’’’’
@@
Ŗ "&#’’’’’’’’’
@Ąś
H__inference_concatenate_layer_call_and_return_conditional_losses_1044381­v¢s
l¢i
gd
0-
inputs/0’’’’’’’’’@
0-
inputs/1’’’’’’’’’ 
Ŗ "3¢0
)&
0’’’’’’’’’`
 Ņ
-__inference_concatenate_layer_call_fn_1044374 v¢s
l¢i
gd
0-
inputs/0’’’’’’’’’@
0-
inputs/1’’’’’’’’’ 
Ŗ "&#’’’’’’’’’`Ā
F__inference_conv3d_10_layer_call_and_return_conditional_losses_1044413x&'=¢:
3¢0
.+
inputs’’’’’’’’’ 
Ŗ "3¢0
)&
0’’’’’’’’’
 
+__inference_conv3d_10_layer_call_fn_1044403k&'=¢:
3¢0
.+
inputs’’’’’’’’’ 
Ŗ "&#’’’’’’’’’Į
E__inference_conv3d_1_layer_call_and_return_conditional_losses_1044451x=¢:
3¢0
.+
inputs’’’’’’’’’ 
Ŗ "3¢0
)&
0’’’’’’’’’ 
 
*__inference_conv3d_1_layer_call_fn_1044441k=¢:
3¢0
.+
inputs’’’’’’’’’ 
Ŗ "&#’’’’’’’’’ Į
E__inference_conv3d_2_layer_call_and_return_conditional_losses_1044546x=¢:
3¢0
.+
inputs’’’’’’’’’`
Ŗ "3¢0
)&
0’’’’’’’’’ 
 
*__inference_conv3d_2_layer_call_fn_1044536k=¢:
3¢0
.+
inputs’’’’’’’’’`
Ŗ "&#’’’’’’’’’ Į
E__inference_conv3d_3_layer_call_and_return_conditional_losses_1044565x !=¢:
3¢0
.+
inputs’’’’’’’’’ 
Ŗ "3¢0
)&
0’’’’’’’’’ 
 
*__inference_conv3d_3_layer_call_fn_1044555k !=¢:
3¢0
.+
inputs’’’’’’’’’ 
Ŗ "&#’’’’’’’’’ æ
E__inference_conv3d_4_layer_call_and_return_conditional_losses_1044470v<¢9
2¢/
-*
inputs’’’’’’’’’
@ 
Ŗ "2¢/
(%
0’’’’’’’’’
@@
 
*__inference_conv3d_4_layer_call_fn_1044460i<¢9
2¢/
-*
inputs’’’’’’’’’
@ 
Ŗ "%"’’’’’’’’’
@@æ
E__inference_conv3d_5_layer_call_and_return_conditional_losses_1044489v<¢9
2¢/
-*
inputs’’’’’’’’’
@@
Ŗ "2¢/
(%
0’’’’’’’’’
@@
 
*__inference_conv3d_5_layer_call_fn_1044479i<¢9
2¢/
-*
inputs’’’’’’’’’
@@
Ŗ "%"’’’’’’’’’
@@Ą
E__inference_conv3d_6_layer_call_and_return_conditional_losses_1044584w"#=¢:
3¢0
.+
inputs’’’’’’’’’
@Ą
Ŗ "2¢/
(%
0’’’’’’’’’
@@
 
*__inference_conv3d_6_layer_call_fn_1044574j"#=¢:
3¢0
.+
inputs’’’’’’’’’
@Ą
Ŗ "%"’’’’’’’’’
@@æ
E__inference_conv3d_7_layer_call_and_return_conditional_losses_1044603v$%<¢9
2¢/
-*
inputs’’’’’’’’’
@@
Ŗ "2¢/
(%
0’’’’’’’’’
@@
 
*__inference_conv3d_7_layer_call_fn_1044593i$%<¢9
2¢/
-*
inputs’’’’’’’’’
@@
Ŗ "%"’’’’’’’’’
@@¾
E__inference_conv3d_8_layer_call_and_return_conditional_losses_1044508u;¢8
1¢.
,)
inputs’’’’’’’’’@ @
Ŗ "2¢/
(%
0’’’’’’’’’@ 
 
*__inference_conv3d_8_layer_call_fn_1044498h;¢8
1¢.
,)
inputs’’’’’’’’’@ @
Ŗ "%"’’’’’’’’’@ æ
E__inference_conv3d_9_layer_call_and_return_conditional_losses_1044527v<¢9
2¢/
-*
inputs’’’’’’’’’@ 
Ŗ "2¢/
(%
0’’’’’’’’’@ 
 
*__inference_conv3d_9_layer_call_fn_1044517i<¢9
2¢/
-*
inputs’’’’’’’’’@ 
Ŗ "%"’’’’’’’’’@ æ
C__inference_conv3d_layer_call_and_return_conditional_losses_1044432x=¢:
3¢0
.+
inputs’’’’’’’’’
Ŗ "3¢0
)&
0’’’’’’’’’ 
 
(__inference_conv3d_layer_call_fn_1044422k=¢:
3¢0
.+
inputs’’’’’’’’’
Ŗ "&#’’’’’’’’’ Ī
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1041586 !B¢?
8¢5
/,
input_1’’’’’’’’’`
p 
Ŗ "3¢0
)&
0’’’’’’’’’ 
 Ī
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1041602 !B¢?
8¢5
/,
input_1’’’’’’’’’`
p
Ŗ "3¢0
)&
0’’’’’’’’’ 
 Ķ
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1043929~ !A¢>
7¢4
.+
inputs’’’’’’’’’`
p 
Ŗ "3¢0
)&
0’’’’’’’’’ 
 Ķ
K__inference_conv_block3d_1_layer_call_and_return_conditional_losses_1043947~ !A¢>
7¢4
.+
inputs’’’’’’’’’`
p
Ŗ "3¢0
)&
0’’’’’’’’’ 
 ¦
0__inference_conv_block3d_1_layer_call_fn_1041495r !B¢?
8¢5
/,
input_1’’’’’’’’’`
p 
Ŗ "&#’’’’’’’’’ ¦
0__inference_conv_block3d_1_layer_call_fn_1041570r !B¢?
8¢5
/,
input_1’’’’’’’’’`
p
Ŗ "&#’’’’’’’’’ „
0__inference_conv_block3d_1_layer_call_fn_1043898q !A¢>
7¢4
.+
inputs’’’’’’’’’`
p 
Ŗ "&#’’’’’’’’’ „
0__inference_conv_block3d_1_layer_call_fn_1043911q !A¢>
7¢4
.+
inputs’’’’’’’’’`
p
Ŗ "&#’’’’’’’’’ Ģ
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1041266}A¢>
7¢4
.+
input_1’’’’’’’’’
@ 
p 
Ŗ "2¢/
(%
0’’’’’’’’’
@@
 Ģ
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1041282}A¢>
7¢4
.+
input_1’’’’’’’’’
@ 
p
Ŗ "2¢/
(%
0’’’’’’’’’
@@
 Ė
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1043805|@¢=
6¢3
-*
inputs’’’’’’’’’
@ 
p 
Ŗ "2¢/
(%
0’’’’’’’’’
@@
 Ė
K__inference_conv_block3d_2_layer_call_and_return_conditional_losses_1043823|@¢=
6¢3
-*
inputs’’’’’’’’’
@ 
p
Ŗ "2¢/
(%
0’’’’’’’’’
@@
 ¤
0__inference_conv_block3d_2_layer_call_fn_1041175pA¢>
7¢4
.+
input_1’’’’’’’’’
@ 
p 
Ŗ "%"’’’’’’’’’
@@¤
0__inference_conv_block3d_2_layer_call_fn_1041250pA¢>
7¢4
.+
input_1’’’’’’’’’
@ 
p
Ŗ "%"’’’’’’’’’
@@£
0__inference_conv_block3d_2_layer_call_fn_1043774o@¢=
6¢3
-*
inputs’’’’’’’’’
@ 
p 
Ŗ "%"’’’’’’’’’
@@£
0__inference_conv_block3d_2_layer_call_fn_1043787o@¢=
6¢3
-*
inputs’’’’’’’’’
@ 
p
Ŗ "%"’’’’’’’’’
@@Ķ
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1041746~"#$%B¢?
8¢5
/,
input_1’’’’’’’’’
@Ą
p 
Ŗ "2¢/
(%
0’’’’’’’’’
@@
 Ķ
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1041762~"#$%B¢?
8¢5
/,
input_1’’’’’’’’’
@Ą
p
Ŗ "2¢/
(%
0’’’’’’’’’
@@
 Ģ
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1043991}"#$%A¢>
7¢4
.+
inputs’’’’’’’’’
@Ą
p 
Ŗ "2¢/
(%
0’’’’’’’’’
@@
 Ģ
K__inference_conv_block3d_3_layer_call_and_return_conditional_losses_1044009}"#$%A¢>
7¢4
.+
inputs’’’’’’’’’
@Ą
p
Ŗ "2¢/
(%
0’’’’’’’’’
@@
 „
0__inference_conv_block3d_3_layer_call_fn_1041655q"#$%B¢?
8¢5
/,
input_1’’’’’’’’’
@Ą
p 
Ŗ "%"’’’’’’’’’
@@„
0__inference_conv_block3d_3_layer_call_fn_1041730q"#$%B¢?
8¢5
/,
input_1’’’’’’’’’
@Ą
p
Ŗ "%"’’’’’’’’’
@@¤
0__inference_conv_block3d_3_layer_call_fn_1043960p"#$%A¢>
7¢4
.+
inputs’’’’’’’’’
@Ą
p 
Ŗ "%"’’’’’’’’’
@@¤
0__inference_conv_block3d_3_layer_call_fn_1043973p"#$%A¢>
7¢4
.+
inputs’’’’’’’’’
@Ą
p
Ŗ "%"’’’’’’’’’
@@Ė
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1041426|@¢=
6¢3
-*
input_1’’’’’’’’’@ @
p 
Ŗ "2¢/
(%
0’’’’’’’’’@ 
 Ė
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1041442|@¢=
6¢3
-*
input_1’’’’’’’’’@ @
p
Ŗ "2¢/
(%
0’’’’’’’’’@ 
 Ź
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1043867{?¢<
5¢2
,)
inputs’’’’’’’’’@ @
p 
Ŗ "2¢/
(%
0’’’’’’’’’@ 
 Ź
K__inference_conv_block3d_4_layer_call_and_return_conditional_losses_1043885{?¢<
5¢2
,)
inputs’’’’’’’’’@ @
p
Ŗ "2¢/
(%
0’’’’’’’’’@ 
 £
0__inference_conv_block3d_4_layer_call_fn_1041335o@¢=
6¢3
-*
input_1’’’’’’’’’@ @
p 
Ŗ "%"’’’’’’’’’@ £
0__inference_conv_block3d_4_layer_call_fn_1041410o@¢=
6¢3
-*
input_1’’’’’’’’’@ @
p
Ŗ "%"’’’’’’’’’@ ¢
0__inference_conv_block3d_4_layer_call_fn_1043836n?¢<
5¢2
,)
inputs’’’’’’’’’@ @
p 
Ŗ "%"’’’’’’’’’@ ¢
0__inference_conv_block3d_4_layer_call_fn_1043849n?¢<
5¢2
,)
inputs’’’’’’’’’@ @
p
Ŗ "%"’’’’’’’’’@ Ģ
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1041872}&'B¢?
8¢5
/,
input_1’’’’’’’’’ 
p 
Ŗ "3¢0
)&
0’’’’’’’’’
 Ģ
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1041881}&'B¢?
8¢5
/,
input_1’’’’’’’’’ 
p
Ŗ "3¢0
)&
0’’’’’’’’’
 Ė
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1043689|&'A¢>
7¢4
.+
inputs’’’’’’’’’ 
p 
Ŗ "3¢0
)&
0’’’’’’’’’
 Ė
K__inference_conv_block3d_5_layer_call_and_return_conditional_losses_1043699|&'A¢>
7¢4
.+
inputs’’’’’’’’’ 
p
Ŗ "3¢0
)&
0’’’’’’’’’
 ¤
0__inference_conv_block3d_5_layer_call_fn_1041817p&'B¢?
8¢5
/,
input_1’’’’’’’’’ 
p 
Ŗ "&#’’’’’’’’’¤
0__inference_conv_block3d_5_layer_call_fn_1041863p&'B¢?
8¢5
/,
input_1’’’’’’’’’ 
p
Ŗ "&#’’’’’’’’’£
0__inference_conv_block3d_5_layer_call_fn_1043670o&'A¢>
7¢4
.+
inputs’’’’’’’’’ 
p 
Ŗ "&#’’’’’’’’’£
0__inference_conv_block3d_5_layer_call_fn_1043679o&'A¢>
7¢4
.+
inputs’’’’’’’’’ 
p
Ŗ "&#’’’’’’’’’Ģ
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1041106B¢?
8¢5
/,
input_1’’’’’’’’’
p 
Ŗ "3¢0
)&
0’’’’’’’’’ 
 Ģ
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1041122B¢?
8¢5
/,
input_1’’’’’’’’’
p
Ŗ "3¢0
)&
0’’’’’’’’’ 
 Ė
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1043743~A¢>
7¢4
.+
inputs’’’’’’’’’
p 
Ŗ "3¢0
)&
0’’’’’’’’’ 
 Ė
I__inference_conv_block3d_layer_call_and_return_conditional_losses_1043761~A¢>
7¢4
.+
inputs’’’’’’’’’
p
Ŗ "3¢0
)&
0’’’’’’’’’ 
 ¤
.__inference_conv_block3d_layer_call_fn_1041015rB¢?
8¢5
/,
input_1’’’’’’’’’
p 
Ŗ "&#’’’’’’’’’ ¤
.__inference_conv_block3d_layer_call_fn_1041090rB¢?
8¢5
/,
input_1’’’’’’’’’
p
Ŗ "&#’’’’’’’’’ £
.__inference_conv_block3d_layer_call_fn_1043712qA¢>
7¢4
.+
inputs’’’’’’’’’
p 
Ŗ "&#’’’’’’’’’ £
.__inference_conv_block3d_layer_call_fn_1043725qA¢>
7¢4
.+
inputs’’’’’’’’’
p
Ŗ "&#’’’’’’’’’ 
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_1044029ø_¢\
U¢R
PM
inputsA’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "U¢R
KH
0A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 į
1__inference_max_pooling3d_1_layer_call_fn_1044024«_¢\
U¢R
PM
inputsA’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "HEA’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_1044019ø_¢\
U¢R
PM
inputsA’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "U¢R
KH
0A’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ß
/__inference_max_pooling3d_layer_call_fn_1044014«_¢\
U¢R
PM
inputsA’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "HEA’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’Š
%__inference_signature_wrapper_1042749¦"#$% !&'I¢F
¢ 
?Ŗ<
:
input_1/,
input_1’’’’’’’’’"AŖ>
<
output_10-
output_1’’’’’’’’’Ś
D__inference_u_net3d_layer_call_and_return_conditional_losses_1042632"#$% !&'B¢?
8¢5
/,
input_1’’’’’’’’’
p 
Ŗ "3¢0
)&
0’’’’’’’’’
 Ś
D__inference_u_net3d_layer_call_and_return_conditional_losses_1042692"#$% !&'B¢?
8¢5
/,
input_1’’’’’’’’’
p
Ŗ "3¢0
)&
0’’’’’’’’’
 Ł
D__inference_u_net3d_layer_call_and_return_conditional_losses_1043254"#$% !&'A¢>
7¢4
.+
inputs’’’’’’’’’
p 
Ŗ "3¢0
)&
0’’’’’’’’’
 Ł
D__inference_u_net3d_layer_call_and_return_conditional_losses_1043661"#$% !&'A¢>
7¢4
.+
inputs’’’’’’’’’
p
Ŗ "3¢0
)&
0’’’’’’’’’
 ²
)__inference_u_net3d_layer_call_fn_1042339"#$% !&'B¢?
8¢5
/,
input_1’’’’’’’’’
p 
Ŗ "&#’’’’’’’’’²
)__inference_u_net3d_layer_call_fn_1042572"#$% !&'B¢?
8¢5
/,
input_1’’’’’’’’’
p
Ŗ "&#’’’’’’’’’±
)__inference_u_net3d_layer_call_fn_1042798"#$% !&'A¢>
7¢4
.+
inputs’’’’’’’’’
p 
Ŗ "&#’’’’’’’’’±
)__inference_u_net3d_layer_call_fn_1042847"#$% !&'A¢>
7¢4
.+
inputs’’’’’’’’’
p
Ŗ "&#’’’’’’’’’Ć
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_1044368s<¢9
2¢/
-*
inputs’’’’’’’’’@ 
Ŗ "3¢0
)&
0’’’’’’’’’
@
 
1__inference_up_sampling3d_1_layer_call_fn_1044254f<¢9
2¢/
-*
inputs’’’’’’’’’@ 
Ŗ "&#’’’’’’’’’
@Į
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_1044249s<¢9
2¢/
-*
inputs’’’’’’’’’
@@
Ŗ "3¢0
)&
0’’’’’’’’’@
 
/__inference_up_sampling3d_layer_call_fn_1044034f<¢9
2¢/
-*
inputs’’’’’’’’’
@@
Ŗ "&#’’’’’’’’’@