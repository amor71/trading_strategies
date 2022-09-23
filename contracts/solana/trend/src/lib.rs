use borsh::{BorshDeserialize, BorshSerialize};
use solana_program::{
    account_info::{next_account_info, AccountInfo},
    entrypoint,
    entrypoint::ProgramResult,
    msg,
    program_error::ProgramError,
    pubkey::Pubkey,
};

/// Define the type of state stored in accounts
#[derive(BorshSerialize, BorshDeserialize, Debug)]
pub struct AssetScore {
    /// asset score
    pub slope: f32,
    pub intercept: f32,
}

// Declare and export the program's entrypoint
entrypoint!(process_instruction);

fn parse_data(data:  &[u8]) -> (Vec<i32>, Vec<i32>) {
    let mut parsed_data = Vec::new();
    let mut parsed_data_frac = Vec::new();
    let mut index : usize = 1;

    let length = if data[0] == 0 {1 + (data.len() - 1) / 2} else {1 + (data.len() - 1) / 3};
    for _i in 1..length {
        let element : i32;
        let element_frac : i32;

        if data[0] == 0 {
            element = data[index] as i32;
            element_frac = data[index + 1] as i32; 
            index = index + 2;
        } else {
            element = (data[index] as i32) * 256 + (data[index+1] as i32);
            element_frac = data[index + 2] as i32; 
            index = index + 3;
        }
         
        parsed_data.push(element);
        parsed_data_frac.push(element_frac);
    }
    //msg!("parsed_data: {:?}", parsed_data);
    (parsed_data, parsed_data_frac) 
}

// Program entrypoint's implementation
pub fn process_instruction(
    program_id: &Pubkey, 
    accounts: &[AccountInfo], 
    instruction_data: &[u8], 
) -> ProgramResult {
    let accounts_iter = &mut accounts.iter();
    let account = next_account_info(accounts_iter)?;

    // The account must be owned by the program in order to modify its data
    if account.owner != program_id {
        msg!("response account does not have the correct program id");
        return Err(ProgramError::IncorrectProgramId);
    }

    let (data, frac_data) = parse_data(instruction_data);

    // calculate log
    let len = data.len();
    let mut sum_x : i32 = 0;
    let mut sum_x_2 : i32 = 0;
    let mut sum_y : i32 = 0;
    let mut frac_sum_y : i32 = 0;
    let mut sum_xy : i32 = 0;
    let mut sum_x_frac_y : i32 = 0;

    let mut x : i32 = 0;
    for i in 0..len {
        sum_x = sum_x + x;
        sum_x_2 = sum_x_2 + x * x;
        sum_xy = sum_xy + x * data[i];
        sum_x_frac_y =  sum_x_frac_y + x * frac_data[i];
        sum_y = sum_y + data[i];
        frac_sum_y = frac_sum_y + frac_data[i];
        x = x + 1;
    }
    
    let denom : i32 = x * sum_x_2 - sum_x * sum_x;
    let real_sum_y : f32 = sum_y as f32 + (frac_sum_y as f32) / 100.;
    let real_sum_xy : f32 = sum_xy as f32 + (sum_x_frac_y as f32) / 100.;
    let slope : f32 = ((x as f32) * real_sum_xy - (sum_x as f32) * real_sum_y) / (denom as f32);
    let intercept : f32 = (real_sum_y * (sum_x_2 as f32) - (sum_x as f32)* real_sum_xy) / (denom as f32);
    //let mean_data : f32 = real_sum_y / (x as f32);

    //let mut tss : f32 = 0.;
    //let mut rss : f32 = 0.;

    //for i in 0..len {
    //    let val : f32 = (data[i] as f32) + (frac_data[i] as f32) / 100.;
    //    let predicted_value : f32 = slope * val + intercept;
    //    rss = rss + (val - predicted_value) * (val  - predicted_value);
    //    tss = tss + (val - mean_data) * (val - mean_data);
    //}

    //let r_2 : f32 = 1.0 ; // if tss != 0. {(tss - rss) / tss} else {1.};
    let mut result = AssetScore::try_from_slice(&account.data.borrow())?;
    result.slope = slope;
    result.intercept = intercept;
    //result.r_2 = r_2;
    result.serialize(&mut &mut account.data.borrow_mut()[..])?;

    Ok(())
}



// Sanity tests
#[cfg(test)]
mod test {
    use super::*;
    use solana_program::clock::Epoch;
    use std::mem;

    #[test]
    fn test_sanity() {
        let program_id = Pubkey::default();
        let key = Pubkey::default();
        let mut lamports = 0;
        let mut data = vec![0; 2*3*mem::size_of::<u32>()];
        let owner = Pubkey::default();
        let account = AccountInfo::new(
            &key,
            false,
            true,
            &mut lamports,
            &mut data,
            &owner,
            false,
            Epoch::default(),
        );
        let instruction_data = [0, 1, 0, 50, 45, 75, 12, 100, 99, 200, 12, 200, 34];
        let accounts = vec![account];

        process_instruction(&program_id, &accounts, &instruction_data).unwrap();
        
        let result = AssetScore::try_from_slice(&accounts[0].data.borrow()).unwrap().slope;
        msg!("Score for {:?} is {}", instruction_data, result);
        
        let instruction_data = [1, 3, 137, 10, 3, 129, 3, 134, 3, 14, 3, 145, 3, 116, 3, 14];

        msg!("length of payload {}", instruction_data.len());
        process_instruction(&program_id, &accounts, &instruction_data).unwrap();
        let result = AssetScore::try_from_slice(&accounts[0].data.borrow()).unwrap().slope;
        msg!("Score for {:?} is {}", instruction_data, result);
     
     
    }
}
