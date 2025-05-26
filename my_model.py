from Aplus.models.transformer import *

class TPM(BaseModel):
    def __init__(self, multi_head, d_model, d_ff, n_output):
        super().__init__()
        self.encoder = EncoderLayer(multi_head, d_model, d_ff, dropout=0)
        self.mapping = nn.Linear(d_model, n_output)
    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1, keepdim=False)
        x = self.mapping(x)
        return x

class TIC(BaseModel):
    def __init__(self, stack, n_input, n_output, multi_head=8, d_model=256, d_ff=512):
        """
        :param stack: 堆叠多少个编码器层
        :param multi_head: 多头注意力头的数量
        :param d_model: 隐藏层的维度
        n_input:  [72] = [6 * (3 + 3*3)]
        n_output: [36] = [6 * 6]
        """
        super().__init__()
        # print n_input, n_output
        print(f"TIC Network: input: {n_input}, output: {n_output}")
    
        self.input_embedding_layer = Embedder(n_input=n_input, d_model=d_model)
        # self.input_pe = PositionalEncoding(d_model)
        self.encoder_stack = []
        self.imu_num = 6
        for i in range(stack):
            encoder_layer = EncoderLayer(multi_head, d_model, d_ff, dropout=0)
            self.encoder_stack.append(encoder_layer)
        self.encoder_banckbone = nn.ModuleList(self.encoder_stack)

        self.TPM_global = TPM(multi_head, d_model, d_ff, n_output)
        self.TPM_local = TPM(multi_head, d_model, d_ff, n_output)


    def forward(self, x):
        x = self.input_embedding_layer(x)
        # x = self.input_pe(x)

        for encoder_layer in self.encoder_banckbone:
            x = encoder_layer(x, None)

        global_shift = self.TPM_global(x)
        local_shift = self.TPM_local(x)

        return global_shift, local_shift


