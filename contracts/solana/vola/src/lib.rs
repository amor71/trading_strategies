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
pub struct AssetVolatility {
    pub vola: f32,
}

// Declare and export the program's entrypoint
entrypoint!(process_instruction);

fn parse_data(data:  &[u8]) -> Vec<i32> {
    let mut parsed_data = Vec::new();
    let mut index : usize = 1;

    let length = if data[0] == 0 {data.len()} else {1 + (data.len() - 1) / 2};
    for _i in 1..length {
        let element : i32;

        if data[0] == 0 {
            element = data[index] as i32; 
            index = index + 1;
        } else {
            element = (data[index] as i32) * 256 + (data[index+1] as i32) ; 
            index = index + 2;
        }
         
        parsed_data.push(element);
    }

    parsed_data
}

// Program entrypoint's implementation
pub fn process_instruction(
    program_id: &Pubkey, 
    accounts: &[AccountInfo], 
    instruction_data: &[u8], 
) -> ProgramResult {
    let windows_size = 20;
    let accounts_iter = &mut accounts.iter();
    let account = next_account_info(accounts_iter)?;

    // The account must be owned by the program in order to modify its data
    if account.owner != program_id {
        msg!("response account does not have the correct program id");
        return Err(ProgramError::IncorrectProgramId);
    }

    let data = parse_data(instruction_data);
    let len = data.len();

    if len < windows_size {
        msg!("Not enough data to calculate volatility");
        return Err(ProgramError::IncorrectProgramId);
    }

    let mut pct_change = Vec::new();
    let mut mean : f32 = 0.;
    let mut sum : f32 = 0.;
    for i in 1..len {
        let y : f32 = (data[i] - data[i -1]) as f32 / data[i-1] as f32;
        pct_change.push(y);

        sum = sum + y;

        if i > windows_size {
            mean = sum / (windows_size as f32);
            sum -= pct_change[i - windows_size];
        }
    }

    sum = 0.;
    for i in (len - windows_size - 1)  .. (len - 1) {
        sum = sum + ((pct_change[i] as f32)- mean) * ((pct_change[i] as f32) - mean);
    }
    sum = sum / (windows_size as f32);


    let mut result = AssetVolatility::try_from_slice(&account.data.borrow())?;
    result.vola = sum.sqrt();
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
        let mut data = vec![0; mem::size_of::<u32>()];
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
        
        let accounts = vec![account];
        let instruction_data = [1, 3, 137, 3, 129, 3, 134, 3, 149, 3, 145, 3, 116, 3, 142, 3, 119, 3, 115, 3, 133, 3, 114, 3, 116, 3, 139, 3, 142, 3, 138, 3, 144, 3, 136, 3, 121, 3, 133, 3, 140, 3, 132, 3, 116, 3, 130, 3, 130, 3, 133, 3, 141, 3, 140, 3, 128, 3, 132, 3, 134, 3, 130, 3, 136, 3, 111, 3, 114, 3, 111, 3, 93, 3, 107, 3, 104, 3, 87, 3, 68, 3, 52, 3, 45, 3, 46, 3, 21, 3, 7, 3, 19, 3, 16, 3, 8, 3, 28, 3, 43, 3, 46, 3, 42, 3, 26, 3, 30, 3, 37, 3, 27, 3, 33, 2, 255, 2, 249, 2, 237, 3, 2, 3, 4, 2, 239, 2, 233, 2, 222, 2, 210, 2, 209, 2, 228, 2, 221, 2, 192, 2, 215, 2, 215, 2, 179, 2, 160, 2, 146, 2, 176, 2, 180, 2, 170, 2, 180, 2, 190, 2, 213, 2, 223, 2, 221, 2, 216, 2, 229, 2, 219, 2, 220, 2, 223, 2, 232, 2, 255, 3, 3, 2, 246, 2, 252, 3, 8, 2, 246, 2, 227, 2, 231, 2, 218, 2, 211, 2, 199, 2, 198, 2, 171, 2, 163, 2, 184, 2, 181, 2, 164, 2, 146, 2, 151, 2, 130, 2, 126, 2, 134, 2, 108, 2, 108, 2, 114, 2, 148, 2, 117, 2, 112, 2, 89, 2, 93, 2, 84, 2, 84, 2, 98, 2, 82, 2, 95, 2, 75, 2, 76, 2, 84, 2, 95, 2, 101, 2, 110, 2, 135, 2, 149, 2, 152, 2, 143, 2, 169, 2, 159, 2, 163, 2, 167, 2, 150, 2, 147, 2, 105, 2, 86, 2, 79, 2, 87, 2, 73, 2, 70, 2, 102, 2, 97, 2, 112, 2, 135, 2, 124, 2, 117, 2, 105, 2, 97, 2, 104, 2, 108, 2, 102, 2, 111, 2, 107, 2, 93, 2, 91, 2, 84, 2, 76, 2, 88, 2, 85, 2, 116, 2, 122, 2, 127, 2, 121, 2, 124, 2, 110, 2, 131, 2, 147, 2, 157, 2, 159, 2, 154, 2, 178, 2, 184, 2, 183, 2, 184, 2, 184, 2, 213, 2, 216, 2, 239, 2, 243, 2, 245, 2, 231, 2, 233, 2, 201, 2, 183, 2, 184, 2, 188, 2, 198, 2, 164, 2, 163, 2, 160, 2, 154, 2, 152, 2, 146];

        msg!("length of payload {}", instruction_data.len());
        process_instruction(&program_id, &accounts, &instruction_data).unwrap();
        let result = AssetVolatility::try_from_slice(&accounts[0].data.borrow()).unwrap().vola;
        msg!("Vola for {:?} is {}", instruction_data, result);
     
     
    }
}
