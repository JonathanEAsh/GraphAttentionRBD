import pandas as pd
import itertools
import statistics
df = pd.read_csv('505_betas.csv')
md_456 = {'456': ['F', 'S', 'I', 'T'], '473': ['Y', 'F', 'H', 'V', 'L', 'D'], '475': ['A', 'D', 'V', 'L', 'H', 'P'], '476': ['G', 'L', 'S', 'W', 'A', 'V'], '485': ['G', 'A', 'V', 'S', 'L', 'W'], '486': ['F', 'I', 'S', 'T']}
md_505 = {'498': ['R', 'Q', 'V', 'L', 'G', 'E'], '499': ['P', 'F', 'Y', 'S', 'L', 'H'], '500': ['T', 'W', 'S', 'R', 'M', 'L'], '501': ['Y', 'T', 'S', 'N', 'H', 'P'], '505': ['H', 'Y', 'F', 'L', 'D', 'V']}
# for a in md_505.keys():
# 	temp = []
# 	for b, c in zip(df['betas'], df['values']):
# 		if a in b:
# 			temp.append(c)
# 	print(a)
# 	print(sum(temp))
new_betas = []
new_vals = []
for a, b in zip(df['betas'], df['values']):
	if a.count('_') == 6:
		# triples - i think im doing this correctly
		pos1_id = a.split('_')[1]+'_'+a.split('_')[2]
		pos1 = a.split('_')[1]
		pos2_id = a.split('_')[3]+'_'+a.split('_')[4]
		pos2 = a.split('_')[3]
		pos3_id = a.split('_')[5]+'_'+a.split('_')[6]
		pos3 = a.split('_')[5]
		means = []
		temp1 = []
		temp2 = []
		temp3 = []
		temp4 = []
		temp5 = []
		temp6 = []
		temp7 = []
		for c, d in zip(df['betas'], df['values']):
			if c.count('_') == 6:
				if pos1_id in c and pos2_id in c and pos3 in c:
					temp1.append(d)
				if pos1_id in c and pos2 in c and pos3_id in c:
					temp2.append(d)
				if pos1 in c and pos2_id in c and pos3_id in c:
					temp3.append(d)
				if pos1 in c and pos2 in c and pos3_id in c:
					temp4.append(d)
				if pos1 in c and pos2_id in c and pos3 in c:
					temp5.append(d)
				if pos1_id in c and pos2 in c and pos3 in c:
					temp6.append(d)
				if pos1 in c and pos2 in c and pos3 in c:
					temp7.append(d)
		new_betas.append(a)
		new_vals.append(b - (statistics.mean(temp1) + statistics.mean(temp2) + statistics.mean(temp3)) + (statistics.mean(temp4) + statistics.mean(temp5) + statistics.mean(temp6)) - statistics.mean(temp7))
	elif a.count('_') == 4:
		# doubles - need to separate triples by third position
		pos1_id = a.split('_')[1]+'_'+a.split('_')[2]
		pos1 = a.split('_')[1]
		pos2_id = a.split('_')[3]+'_'+a.split('_')[4]
		pos2 = a.split('_')[3]
		# 3 long
		# 4 for 456
		other_positions = [z for z in md_505.keys() if z != pos1 and z != pos2]
		means = []
		temp1 = []
		temp2 = []
		temp3 = []
		temp4 = []
		temp5 = []
		temp6 = []
		temp7 = []
		temp8 = []
		temp9 = []
		temp10 = []
		temp11 = []
		temp12 = []
		temp13 = []
		temp14 = []
		temp15 = []
		temp16 = []
		temp17 = []
		temp18 = []
		temp19 = []
		for c, d in zip(df['betas'], df['values']):
			if c.count('_') == 4:
				# double queries
				if pos1_id in c and pos2 in c:
					temp1.append(d)
				if pos1 in c and pos2_id in c:
					temp2.append(d)
				if pos1 in c and pos2 in c:
					temp3.append(d)
			elif c.count('_') == 6:
				# triple queries
				# separate by other positions
				if pos1_id in c and pos2_id in c and other_positions[0] in c:
					temp4.append(d)
				if pos1_id in c and pos2 in c and other_positions[0] in c:
					temp5.append(d)
				if pos1 in c and pos2_id in c and other_positions[0] in c:
					temp6.append(d)
				if pos1 in c and pos2 in c and other_positions[0] in c:
					temp7.append(d)
				if pos1_id in c and pos2_id in c and other_positions[1] in c:
					temp8.append(d)
				if pos1_id in c and pos2 in c and other_positions[1] in c:
					temp9.append(d)
				if pos1 in c and pos2_id in c and other_positions[1] in c:
					temp10.append(d)
				if pos1 in c and pos2 in c and other_positions[1] in c:
					temp11.append(d)
				if pos1_id in c and pos2_id in c and other_positions[2] in c:
					temp12.append(d)
				if pos1_id in c and pos2 in c and other_positions[2] in c:
					temp13.append(d)
				if pos1 in c and pos2_id in c and other_positions[2] in c:
					temp14.append(d)
				if pos1 in c and pos2 in c and other_positions[2] in c:
					temp15.append(d)
				# only for 456
				# if pos1_id in c and pos2_id in c and other_positions[3] in c:
				# 	temp16.append(d)
				# if pos1_id in c and pos2 in c and other_positions[3] in c:
				# 	temp17.append(d)
				# if pos1 in c and pos2_id in c and other_positions[3] in c:
				# 	temp18.append(d)
				# if pos1 in c and pos2 in c and other_positions[3] in c:
				# 	temp19.append(d)
		new_betas.append(a)
		new_vals.append(b - (statistics.mean(temp1) + statistics.mean(temp2) - statistics.mean(temp3)) + (statistics.mean(temp4) - statistics.mean(temp5) - statistics.mean(temp6) + statistics.mean(temp7)) + (statistics.mean(temp8) - statistics.mean(temp9) - statistics.mean(temp10) - statistics.mean(temp11)) \
			+ (statistics.mean(temp12) - statistics.mean(temp13) - statistics.mean(temp14) - statistics.mean(temp15))) \
			#+ (statistics.mean(temp16) - statistics.mean(temp17) - statistics.mean(temp18) + statistics.mean(temp19)))
	else:
		# singles
		pos1_id = a.split('_')[1]+'_'+a.split('_')[2]
		pos1 = a.split('_')[1]
		# 4 long
		# 5 for 456
		other_positions = [z for z in md_505.keys() if z != pos1]
		temp1 = []
		temp2 = []
		temp3 = []
		temp4 = []
		temp5 = []
		temp6 = []
		temp7 = []
		temp8 = []
		temp9 = []
		temp10 = []
		temp11 = []
		temp12 = []
		temp13 = []
		temp14 = []
		temp15 = []
		temp16 = []
		temp17 = []
		temp18 = []
		temp19 = []
		temp20 = []
		temp21 = []
		temp22 = []
		temp23 = []
		temp24 = []
		temp25 = []
		temp26 = []
		temp27 = []
		temp28 = []
		temp29 = []
		temp30 = []
		temp31 = []
		temp32 = []
		temp33 = []
		temp34 = []
		temp35 = []
		temp36 = []
		temp37 = []
		temp38 = []
		temp39 = []
		for c, d in zip(df['betas'], df['values']):
			if c.count('_') == 2:
				# single queries
				if pos1 in c:
					temp1.append(d)
			if c.count('_') == 4:
				# double queries
				if pos1_id in c and other_positions[0] in c:
					temp2.append(d)
				if pos1 in c and other_positions[0] in c:
					temp3.append(d)
				if pos1_id in c and other_positions[1] in c:
					temp4.append(d)
				if pos1 in c and other_positions[1] in c:
					temp5.append(d)
				if pos1_id in c and other_positions[2] in c:
					temp6.append(d)
				if pos1 in c and other_positions[2] in c:
					temp7.append(d)
				if pos1_id in c and other_positions[3] in c:
					temp8.append(d)
				if pos1 in c and other_positions[3] in c:
					temp9.append(d)
				# only 456
				# if pos1_id in c and other_positions[4] in c:
				# 	temp10.append(d)
				# if pos1 in c and other_positions[4] in c:
				# 	temp11.append(d)
			elif c.count('_') == 6:
				# triple queries
				if pos1_id in c and other_positions[0] in c and other_positions[1] in c:
					temp10.append(d)
				if pos1 in c and other_positions[0] in c and other_positions[1] in c:
					temp11.append(d)
				if pos1_id in c and other_positions[0] in c and other_positions[2] in c:
					temp12.append(d)
				if pos1 in c and other_positions[0] in c and other_positions[2] in c:
					temp13.append(d)
				if pos1_id in c and other_positions[0] in c and other_positions[3] in c:
					temp14.append(d)
				if pos1 in c and other_positions[0] in c and other_positions[3] in c:
					temp15.append(d)
				# if pos1_id in c and other_positions[0] in c and other_positions[4] in c:
				# 	temp16.append(d)
				# if pos1 in c and other_positions[0] in c and other_positions[4] in c:
				# 	temp17.append(d)
				
				if pos1_id in c and other_positions[1] in c and other_positions[2] in c:
					temp20.append(d)
				if pos1 in c and other_positions[1] in c and other_positions[2] in c:
					temp21.append(d)
				if pos1_id in c and other_positions[1] in c and other_positions[3] in c:
					temp22.append(d)
				if pos1 in c and other_positions[1] in c and other_positions[3] in c:
					temp23.append(d)
				# if pos1_id in c and other_positions[1] in c and other_positions[4] in c:
				# 	temp24.append(d)
				# if pos1 in c and other_positions[1] in c and other_positions[4] in c:
				# 	temp25.append(d)
				
				if pos1_id in c and other_positions[2] in c and other_positions[3] in c:
					temp28.append(d)
				if pos1 in c and other_positions[2] in c and other_positions[3] in c:
					temp29.append(d)
				# if pos1_id in c and other_positions[2] in c and other_positions[4] in c:
				# 	temp30.append(d)
				# if pos1 in c and other_positions[2] in c and other_positions[4] in c:
				# 	temp31.append(d)
				
				# if pos1_id in c and other_positions[3] in c and other_positions[4] in c:
				# 	temp34.append(d)
				# if pos1 in c and other_positions[3] in c and other_positions[4] in c:
				# 	temp35.append(d)
				
		new_betas.append(a)
		# 456
		# new_vals.append(b - (statistics.mean(temp1)) + (statistics.mean(temp2) - statistics.mean(temp3)) + (statistics.mean(temp4) - statistics.mean(temp5)) + (statistics.mean(temp6) - statistics.mean(temp7))  + (statistics.mean(temp8) - statistics.mean(temp9)) \
		# + (statistics.mean(temp12) - statistics.mean(temp13)) + (statistics.mean(temp14) - statistics.mean(temp15)) + (statistics.mean(temp16) - statistics.mean(temp17)) \
		#   + (statistics.mean(temp20) - statistics.mean(temp21))  + (statistics.mean(temp22) - statistics.mean(temp23)) + (statistics.mean(temp24) - statistics.mean(temp25)) + (statistics.mean(temp28) - statistics.mean(temp29)) \
		#    + (statistics.mean(temp30) - statistics.mean(temp31)) + (statistics.mean(temp34) - statistics.mean(temp35)))
		# 505
		
		new_vals.append(b - (statistics.mean(temp1)) + (statistics.mean(temp2) - statistics.mean(temp3)) + (statistics.mean(temp4) - statistics.mean(temp5)) + (statistics.mean(temp6) - statistics.mean(temp7))  + (statistics.mean(temp8) - statistics.mean(temp9)) \
		+ (statistics.mean(temp12) - statistics.mean(temp13)) + (statistics.mean(temp14) - statistics.mean(temp15)) \
		  + (statistics.mean(temp20) - statistics.mean(temp21))  + (statistics.mean(temp22) - statistics.mean(temp23)) + (statistics.mean(temp28) - statistics.mean(temp29)))
print(len(new_betas))
print(len(df['betas']))
df2 = pd.DataFrame(zip(new_betas, new_vals), columns=['betas', 'values'])
df2.to_csv('505_betas_norm.csv',index=False)
# print(len(temp1))
# print(len(temp2))
# print(len(temp3))
# print(len(temp4))
# print(len(temp5))
# print(len(temp6))
# print(len(temp7))
# print(len(temp8))
# print(len(temp9))
# print(len(temp12))
# print(len(temp13))
# print(len(temp14))
# print(len(temp15))
# print(len(temp20))
# print(len(temp21))
# print(len(temp22))
# print(len(temp23))
# print(len(temp28))
# print(len(temp29))