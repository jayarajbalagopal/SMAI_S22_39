####
# 
#
#

# Dependencies

base_path = "/mnt/img2mol_val_results/"

total_match = 0
total_mse = 0

for it in range(0, 39):

    

    file1 = open(base_path+"match_"+str(it)+".txt")
    match = file1.read()
    total_match = total_match + int(match)
    file1.close()

    print("batch:", it, "match:", match)

    file2 = open(base_path+"mse_"+str(it)+".txt")
    total_mse = total_mse + float(file2.read())*128
    file2.close()

total_mse = total_mse / 5000

print("total matches:", total_match)
print("total_mse:", total_mse)
