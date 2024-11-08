import torch


class PageAllocator():

    def __init__(self, page_num, page_size, device):
        self.page_num = page_num
        self.page_size = page_size
        self.device = device
        self.req2page = {}
        self.req2last_len = {}
        self.req2last_page = {}

        # temporary
        self.next_page_id = 0

    def alloc(self, req_id, req_len):
        if req_id in self.req2last_len:
            last_page_len = self.req2last_len[req_id]
            req_len -= self.page_size - last_page_len
            reg_has_page = True
        else:
            reg_has_page = False

        need_page_num = (req_len + self.page_size - 1) // self.page_size
        last_page_len = req_len % self.page_size
        if last_page_len == 0:
            last_page_len = self.page_size
        if self.next_page_id + need_page_num > self.page_num:
            raise ValueError(
                "Page allocator out of memory, please increase page_num")

        if not reg_has_page:
            self.req2page[req_id] = torch.arange(self.next_page_id,
                                                 self.next_page_id +
                                                 need_page_num,
                                                 dtype=torch.int32,
                                                 device=self.device)
            self.req2last_len[req_id] = last_page_len
        else:
            self.req2page[req_id] = torch.cat([
                self.req2page[req_id],
                torch.arange(self.next_page_id,
                             self.next_page_id + need_page_num,
                             dtype=torch.int32,
                             device=self.device)
            ])
            self.req2last_len[req_id] = last_page_len

        self.next_page_id += need_page_num

    def get_metadata(self, req_ids):
        indptr = [0]
        indices = []
        last_lens = []
        last_ptr = 0
        for id in req_ids:
            reg_pages = self.req2page[id]
            last_ptr += len(reg_pages)
            indptr.append(last_ptr)
            indices.append(reg_pages)
            last_lens.append(self.req2last_len[id])
        indices = torch.cat(indices)
        last_lens = torch.tensor(last_lens,
                                 dtype=torch.int32,
                                 device=self.device)
        indtpr = torch.tensor(indptr, dtype=torch.int32, device=self.device)
        return indtpr, indices, last_lens

    def reset(self):
        self.req2page = {}
        self.req2last_len = {}
        self.next_page_id = 0


if __name__ == "__main__":
    alloc = PageAllocator(1000, 32, "cuda:0")
    alloc.alloc(0, 100)
    indptr, indices, last_lens = alloc.get_append_metadata([0])
    print(indptr)
    print(indices)
    print(last_lens)
    print()
    alloc.alloc(0, 156)
    indptr, indices, last_lens = alloc.get_append_metadata([0])
    print(indptr)
    print(indices)
    print(last_lens)
    print()
    alloc.alloc(1, 33)
    indptr, indices, last_lens = alloc.get_append_metadata([0, 1])
    print(indptr)
    print(indices)
    print(last_lens)
    print()
    alloc.alloc(2, 33)
    indptr, indices, last_lens = alloc.get_append_metadata([0, 1, 2])
    print(indptr)
    print(indices)
    print(last_lens)
    print()
    alloc.alloc(1, 63)
    indptr, indices, last_lens = alloc.get_append_metadata([0, 1, 2])
    print(indptr)
    print(indices)
    print(last_lens)
    print()
