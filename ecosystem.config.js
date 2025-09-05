module.exports = {
  apps : [{
    name   : "captcha-ocr",
    script : "server.py",
    exec_mode: "cluster",
    instances: 4,
    args: "",
    interpreter: "/root/miniconda3/envs/captcha/bin/python"
  }
]
}
