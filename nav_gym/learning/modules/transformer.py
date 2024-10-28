import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------------------------------------  Full Transformer Model  -----------------------------------------------------

class Transformer(nn.Module):
    """
    Full Transformer Model with Encoder and Decoder.
    """
    def __init__(self, args):
        super(Transformer, self).__init__()
        # Model parameters
        self.embed_dim = args["embed_dim"]
        self.num_input_tokens = args["num_input_tokens"]
        self.num_output_tokens = args["num_output_tokens"]
        self.out_dim = args["out_dim"]
        
        # Encoder
        encoder_args = {
            "depth": args["depth"],
            "n_heads": args["n_heads"],
            "entity_dim": self.embed_dim,
            "num_input_tokens": self.num_input_tokens,
            "out_dim": self.embed_dim,  # Encoder outputs embeddings of size embed_dim
        }
        self.encoder = TransformerEncoder(encoder_args)
        
        # Decoder
        decoder_args = {
            "depth": args["depth"],
            "n_heads": args["n_heads"],
            "embed_dim": self.embed_dim,
            "num_output_tokens": self.num_output_tokens,
            "out_dim": self.out_dim,
        }
        self.decoder = TransformerDecoder(decoder_args)
        
        
    def forward(self, src_embed, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass for the full Transformer model.
        :param src: Source input (batch_size, num_input_tokens, embed_dim)
        :param tgt: Target input (batch_size, num_output_tokens, embed_dim)
        :param src_mask: Optional mask for the source input
        :param tgt_mask: Optional mask for the target input
        :return: decoder_output (batch_size, num_output_tokens, out_dim)
        """
        # src: (batch_size, num_input_tokens, embed_dim)
        # tgt: (batch_size, num_output_tokens, embed_dim)
        
        # Embedding source input
        encoder_output = self.encoder(src_embed)  # (batch_size, num_input_tokens + 1, embed_dim)
        
        # Remove the CLS token from encoder output if not needed
        encoder_output = encoder_output[:, 1:, :]  # (batch_size, num_input_tokens, embed_dim)
        
        # Run the decoder
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask=tgt_mask)#(batch_size, num_output_tokens, out_dim)
        
        return decoder_output

# ----------------------------------------------  Transformer Encoder Part  -----------------------------------------------------
class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    """
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        # Model parameters
        self.depth = args["depth"]
        self.n_heads = args["n_heads"]
        self.embed_dim = args["entity_dim"]
        self.num_input_tokens = args["num_input_tokens"]
        self.out_dim = args["out_dim"]
        # Positional Encoding
        self.pos_encoding = StandardPositionalEncoding(self.embed_dim, dropout=0.1, max_len=5000)
        # Transformer Encoder Blocks
        self.transformer_encoder_blocks = nn.ModuleList()
        for _ in range(self.depth):
            self.transformer_encoder_blocks.append(TransformerEncoderBlock(dim=self.embed_dim, n_heads=self.n_heads))
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
    
    def forward(self, x, mask=None):
        """
        Forward pass for the Transformer Encoder.
        :param x: Source input embeddings (batch_size, num_input_tokens, embed_dim)
        :param mask: Optional mask for the source input
        :return: Encoder output (batch_size, num_input_tokens + 1, embed_dim)
        """
        batch_size = x.size(0)
        cls_token = torch.zeros(batch_size, 1, self.embed_dim, device=x.device)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (batch_size, num_input_tokens + 1, embed_dim)
        x = self.pos_encoding(x)
        for encoder_block in self.transformer_encoder_blocks:
            x = encoder_block(x, mask=mask)
        x = self.norm(x)
        return x  # Return all token embeddings

class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        hidden_features = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Self_Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim
        )

    def forward(self, x, mask=None):
        """
        Forward pass for the Transformer Encoder Block.
        :param x: Input tensor (batch_size, seq_len, dim)
        :param mask: Optional mask tensor
        :return: Output tensor (batch_size, seq_len, dim)
        """
        #x: (batch_size, num_input_tokens + 1, embed_dim)
        x = x + self.attn(self.norm1(x), mask=mask)#(batch_size, num_input_tokens + 1, embed_dim)
        x = x + self.mlp(self.norm2(x))#(batch_size, num_input_tokens + 1, embed_dim)
        return x

# The Self_Attention and MLP classes are the same as before
class Self_Attention(nn.Module):
    """
    Multi-Head Self-Attention with optional masking.
    """
    def __init__(self, dim, n_heads, qkv_bias=True, attn_p=0., proj_p=0., is_causal=False):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dim_oneHead = dim // n_heads
        self.head_scale = self.dim_oneHead ** -0.5
        self.is_causal = is_causal
        # Linear projections
        self.W_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x, mask=None):
        """
        Forward pass for Self-Attention.
        :param x: Input tensor (batch_size, seq_len, dim)
        :param mask: Optional mask tensor (batch_size, seq_len, seq_len)
        :return: Output tensor after self-attention
        """
        batch_size, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError("Input dimension does not match model dimension.")
        # Linear projections
        # seq_len = num_input_tokens + 1
        #x: (batch_size, num_input_tokens + 1, embed_dim)
        Q = self.W_q(x)#(batch_size, num_input_tokens + 1, embed_dim)
        K = self.W_k(x)#(batch_size, num_input_tokens + 1, embed_dim)
        V = self.W_v(x)# (batch_size, num_input_tokens + 1, embed_dim)
        # Reshape and transpose for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.dim_oneHead).transpose(1, 2)#(batch_size, n_heads, num_input_tokens + 1, dim_oneHead)
        K = K.view(batch_size, seq_len, self.n_heads, self.dim_oneHead).transpose(1, 2)#(batch_size, n_heads, num_input_tokens + 1, dim_oneHead)
        V = V.view(batch_size, seq_len, self.n_heads, self.dim_oneHead).transpose(1, 2)#(batch_size, n_heads, num_input_tokens + 1, dim_oneHead)
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.head_scale #(batch_size, n_heads, num_input_tokens + 1, num_input_tokens + 1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf')) #(batch_size, n_heads, num_input_tokens + 1, num_input_tokens + 1)
        if self.is_causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)#(batch_size, n_heads, num_input_tokens + 1, num_input_tokens + 1)
        attn_weights = self.attn_drop(attn_weights)#(batch_size, n_heads, num_input_tokens + 1, num_input_tokens + 1)
        # Compute context vector
        context = torch.matmul(attn_weights, V)#(batch_size, n_heads, num_input_tokens + 1, dim_oneHead)
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)#(batch_size, num_input_tokens + 1, embed_dim)
        # Final linear projection
        x = self.proj(context)#(batch_size, num_input_tokens + 1, embed_dim)
        x = self.proj_drop(x)#(batch_size, num_input_tokens + 1, embed_dim)
        return x

class MLP(nn.Module):
    """
    Multilayer Perceptron (Feedforward Neural Network).
    """
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """
        Forward pass for the MLP.
        :param x: Input tensor (batch_size, seq_len, in_features)
        :return: Output tensor (batch_size, seq_len, out_features)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ----------------------------------------------  Transformer Decoder Part  -----------------------------------------------------
class TransformerDecoder(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, args):
        super(TransformerDecoder, self).__init__()
        # Model parameters
        self.depth = args["depth"]
        self.n_heads = args["n_heads"]
        self.embed_dim = args["embed_dim"]
        self.num_output_tokens = args["num_output_tokens"]
        self.out_dim = args["out_dim"]
        # Positional Encoding
        self.pos_encoding = StandardPositionalEncoding(self.embed_dim, dropout=0.1, max_len=5000)
        # Transformer Decoder Blocks
        self.transformer_decoder_blocks = nn.ModuleList()
        for _ in range(self.depth):
            self.transformer_decoder_blocks.append(TransformerDecoderBlock(dim=self.embed_dim, n_heads=self.n_heads))
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.head = nn.Linear(self.embed_dim, self.out_dim)

    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        """
        Forward pass for the Transformer Decoder.
        :param x: Target sequence input (batch_size, target_seq_len)
        :param encoder_output: Encoder output (batch_size, source_seq_len, embed_dim)
        :param tgt_mask: Mask for the target sequence (optional)
        :param memory_mask: Mask for the encoder output (optional)
        :return: Predicted logits or outputs
        """
        # x: (batch_size, target_seq_len)
        x = self.pos_encoding(x)
        for decoder_block in self.transformer_decoder_blocks:
            x = decoder_block(x, encoder_output, tgt_mask=tgt_mask, memory_mask=memory_mask)
        x = self.norm(x)
        output = self.head(x)  # (batch_size, target_seq_len, out_dim)
        return output

class TransformerDecoderBlock(nn.Module):
    """
    Transformer Decoder Block containing masked self-attention, cross-attention, and MLP.
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        hidden_features = int(dim * mlp_ratio)
        # Masked Self-Attention
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.self_attn = Self_Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p,
            is_causal=True  # Enable causal masking
        )
        # Cross-Attention
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attn = Cross_Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        # Feedforward MLP
        self.norm3 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim
        )

    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        """
        Forward pass for the Transformer Decoder Block.
        :param x: Target sequence embeddings (batch_size, target_seq_len, embed_dim)
        :param encoder_output: Encoder output (batch_size, source_seq_len, embed_dim)
        :param tgt_mask: Mask for the target sequence (optional)
        :param memory_mask: Mask for the encoder output (optional)
        :return: Updated target sequence embeddings
        """
        # x: (batch_size, target_seq_len, embed_dim)
        if x.device != encoder_output.device:
            x = x.to(encoder_output.device)
        # Masked Self-Attention
        x = x + self.self_attn(self.norm1(x), mask=tgt_mask)
        # Cross-Attention
        x = x + self.cross_attn(self.norm2(x), encoder_output, mask=memory_mask)
        # Feedforward MLP
        x = x + self.mlp(self.norm3(x))
        return x

class Cross_Attention(nn.Module):
    """
    Multi-Head Cross-Attention between decoder output and encoder output.
    """
    def __init__(self, dim, n_heads, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dim_oneHead = dim // n_heads
        self.head_scale = self.dim_oneHead ** -0.5
        # Linear projections
        self.W_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x, encoder_output, mask=None):
        """
        Forward pass for Cross-Attention.
        :param x: Decoder input tensor (batch_size, target_seq_len, dim)
        :param encoder_output: Encoder output tensor (batch_size, source_seq_len, dim)
        :param mask: Optional mask tensor (batch_size, target_seq_len, source_seq_len)
        :return: Output tensor after cross-attention
        """
        batch_size, target_seq_len, dim = x.shape
        source_seq_len = encoder_output.size(1)
        if dim != self.dim:
            raise ValueError("Input dimension does not match model dimension.")
        # Linear projections
        Q = self.W_q(x)#(batch_size, target_seq_len, dim)
        K = self.W_k(encoder_output)#(batch_size, source_seq_len, dim)
        V = self.W_v(encoder_output)#(batch_size, source_seq_len, dim)
        # Reshape and transpose for multi-head attention
        Q = Q.view(batch_size, target_seq_len, self.n_heads, self.dim_oneHead).transpose(1, 2)#(batch_size, n_heads, target_seq_len, dim_oneHead)
        K = K.view(batch_size, source_seq_len, self.n_heads, self.dim_oneHead).transpose(1, 2)#(batch_size, n_heads, source_seq_len, dim_oneHead)
        V = V.view(batch_size, source_seq_len, self.n_heads, self.dim_oneHead).transpose(1, 2)#(batch_size, n_heads, source_seq_len, dim_oneHead)
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.head_scale#(batch_size, n_heads, target_seq_len, source_seq_len)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)#(batch_size, n_heads, target_seq_len, source_seq_len)
        attn_weights = self.attn_drop(attn_weights)#(batch_size, n_heads, target_seq_len, source_seq_len)
        # Compute context vector
        context = torch.matmul(attn_weights, V)#(batch_size, n_heads, target_seq_len, dim_oneHead)
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, target_seq_len, dim)#(batch_size, target_seq_len, dim)
        # Final linear projection
        x = self.proj(context)# (batch_size, target_seq_len, dim)
        x = self.proj_drop(x)# (batch_size, target_seq_len, dim)
        return x

# The StandardPositionalEncoding class remains the same
class StandardPositionalEncoding(nn.Module):
    """
    Positional Encoding using sine and cosine functions.
    """
    def __init__(self, embed_dim, dropout, max_len=5000):
        super(StandardPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)#initiate dropout layer
        # Compute positional encodings once in log space
        
        # compute the positional encoding and store it in the pe tensor
        pe = torch.zeros(max_len, embed_dim)#Dim:(max_len , embed_dim) create a max_len x embed_dim all-zero tensor
        token_pos = torch.arange(0, max_len).unsqueeze(1) # generate an integer sequence from 0 to max_len-1 and add a dimension
        # calculate div_term for scaling the sine and cosine functions of different positions
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        
        # generate the positional encoding using sine and cosine functions, using the sine function for even indices of d_model and the cosine function for odd indices
        pe[:, 0::2] = torch.sin(token_pos * div_term)#Dim:(max_len , embed_dim)
        pe[:, 1::2] = torch.cos(token_pos * div_term)#Dim:(max_len , embed_dim)
        pe = pe.unsqueeze(0)# add a dimension in the first dimension for batch processing
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input tensor.
        :param x: Input tensor (batch_size, seq_len, embed_dim)
        :return: Tensor with positional encoding added
        """
        #x: (n_samples, n_tokens, embed_dim)
        # add the input x to the corresponding positional encoding
        x = x + self.pe[:, :x.size(1)].to(x.device)
        # apply the dropout layer and return the result
        return self.dropout(x)

#---------------------------------------------------Test the model-----------------------------------------------------
if __name__ == "__main__":
    # Define the model parameters
    args = {
        "depth": 2,
        "n_heads": 4,
        "embed_dim": 64,
        "num_input_tokens": 10,
        "num_output_tokens": 12,
        "out_dim": 64,
    }

    # Instantiate the full Transformer model
    model = Transformer(args)

    # Create sample input embeddings
    batch_size = 8
    num_input_tokens = args["num_input_tokens"]
    num_output_tokens = args["num_output_tokens"]
    embed_dim = args["embed_dim"]

    # Source input embeddings
    src_input_embeddings = torch.randn(batch_size, num_input_tokens, embed_dim)

    # Target input embeddings
    tgt_input_embeddings = torch.randn(batch_size, num_output_tokens, embed_dim)

    # Run the model
    output = model(src_input_embeddings, tgt_input_embeddings)

    # Print the output shape
    print("Transformer Output shape:", output.shape)  # Should be (batch_size, num_output_tokens, vocab_size)