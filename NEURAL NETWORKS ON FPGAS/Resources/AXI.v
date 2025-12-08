AXI - Advanced Xtensible Interface  -> memory management like ram(sram,bram)
write and read data 
address 
mem block (RAM) <- read_addr
                -> read_data  
                <- wrie_addr
                <- write_data
                <- write_valid

AXI - general interface to a RAM -> Bursts 
Handshaking 

AXI             G1 read_addr
                G2 read_data  
                G3 wrie_addr
                G4 write_data
                G5 write_valid


                1.) READ BURSTS 
                0,4,8 ---- 36 } 10 contiguous 4 byte packets -> incremental burst --> 4 bytes/transfer
                G1 read_addr -> ar_ready
                             <- ar_valid 
                             <- ar_burst - 2'b01
                             <- ar_addr
                             <- ar_len = 3'b010(no. of transfers)
                             <- ar_size = 4 = 2 ^ ar_len

                G2 read_data <- r_ready 
                             -> r_valid
                             -> read_data
                             -> r_response
                             -> r_last 
                             

               2.) Write Bursts 
                G3 write_addr <- aw_burst --2'b01
                             <- aw_addr 
                             <- aw_len (no. of transfers) 
                             <- aw_size =2^aw_len
                             <- aw_valid
                             -> aw_ready

                G4 write_data <- write_data
                              <- w_strb --> data strobing (0xffffffff) -> AXI lite protocol
                              <- w_last 
                              -> w_ready 
                              <- w_valid 

                G5 write_response -> b_response
                                  <- b_ready
                                  -> b_valid
                
