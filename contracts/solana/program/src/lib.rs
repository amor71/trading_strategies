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
    pub score: f32,
}

// Declare and export the program's entrypoint
entrypoint!(process_instruction);

const fn num_bits<T>() -> usize { std::mem::size_of::<T>() * 8 }

fn log_2(x: i32) -> u32 {
    assert!(x > 0);
    num_bits::<i32>() as u32 - x.leading_zeros() - 1
}

// Program entrypoint's implementation
pub fn process_instruction(
    program_id: &Pubkey, // Public key of the account the hello world program was loaded into
    accounts: &[AccountInfo], // The account to say hello to
    instruction_data: &[u8], // Ignored, all helloworld instructions are hellos
) -> ProgramResult {
    let accounts_iter = &mut accounts.iter();
    let account = next_account_info(accounts_iter)?;

    // The account must be owned by the program in order to modify its data
    if account.owner != program_id {
        msg!("response account does not have the correct program id");
        return Err(ProgramError::IncorrectProgramId);
    }

    // calculate log
    let len = instruction_data.len();
    let mut ln = Vec::new();
    let mut sum_x : i32 = 0;
    let mut sum_x_2 : i32 = 0;
    let mut sum_y : i32 = 0;
    let mut sum_xy : i32 = 0;

    let mut x : i32 = 1;
    for i in 0..len {
        let y : i32 = log_2(instruction_data[i].into()) as i32;
        ln.push(y);

        sum_x = sum_x + x;
        sum_x_2 = sum_x_2 + x * x;
        sum_xy = sum_xy + x * y;
        sum_y = sum_y + y;

        x = x + 1;
    }

    x = x - 1;
    let doneom : f32 = (x * sum_x_2 - sum_x * sum_x) as f32;
    let slope : f32 = ((x as f32) * (sum_xy as f32) - (sum_x as f32) * (sum_y as f32)) / doneom;
    let intercept : f32 = ((sum_y as f32) * (sum_x_2 as f32) - (sum_x as f32) * (sum_xy as f32)) / doneom;

    let mean_data : f32 = (sum_y as f32) / (x as f32);

    let mut tss : f32 = 0.;
    let mut rss : f32 = 0.;

    for i in 0..len {
        let predicted_value : f32 = slope * (ln[i] as f32) + intercept;

        rss = rss + (ln[i] as f32 - predicted_value) * (ln[i] as f32 - predicted_value);
        tss = tss + (ln[i] as f32 - mean_data) * (ln[i] as f32 - mean_data);
    }
    let r : f32 = (tss - rss) / tss;

    let mut result = AssetScore::try_from_slice(&account.data.borrow())?;
    result.score = slope * r * r;
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
        let instruction_data = [1, 50, 75, 100, 200, 200];
        let accounts = vec![account];

        process_instruction(&program_id, &accounts, &instruction_data).unwrap();
        let result = AssetScore::try_from_slice(&accounts[0].data.borrow()).unwrap().score;
        msg!("Score for {:?} is {}", instruction_data, result);
        
     
    }
}
