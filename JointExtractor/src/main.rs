use std::process::Command;
use std::thread;
use std::fs;

fn main() {
    let python_file = "pose_extractor.py";
    // paths to each class
    let classes: Vec<_> = fs::read_dir("input_dataset")
                            .expect("The directory cannot be accesed")
                            .filter_map(|entry|{
                                let path = entry.ok()?.path();
                                // verify that path aims to a directory!
                                if path.is_dir(){
                                    Some(path)
                                } else{
                                    None
                                }
                            }).collect();       
    // Create a vector to handle each threads
    // An iterator that pass through classes will be used too
    let execute_threads: Vec<_> = classes.into_iter().map(|f|{
        let f_clone = f.clone();
        thread::spawn(move || {
            let output = Command::new("python3")
                                        .arg(python_file)
                                        .arg(f)
                                        .output()
                                        .expect("Failed to execute python script");
        println!("Output for {}: {:?}", f_clone.display(), String::from_utf8_lossy(&output.stdout));
        })
    }).collect();
    for handle in execute_threads{
        handle.join().expect("Thread panicked");
    }
}
