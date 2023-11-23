from scripts.robust_cmax import RobustCMax # type: ignore

start                = 0  # time in second
finish               = 10 # time in second
seq                  = "event_file_name" #(without extension)
event_path           = f"./data/{seq}.es"

vx, vy, warped_image = RobustCMax.run(event_path, start, finish)

warped_image.save(f"./img/{seq}_vx_{vx*1e6:.2f}_vy_{vy*1e6:.2f}.png")
warped_image.show()
